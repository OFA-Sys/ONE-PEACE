# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size

        if (w <= h and w == size) or (h <= w and h == size):
            if max_size is not None:
                max_size = int(max_size)
                h = min(h, max_size)
                w = min(w, max_size)
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        if max_size is not None:
           max_size = int(max_size)
           oh = min(oh, max_size)
           ow = min(ow, max_size)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size, interpolation=Image.BICUBIC)

    if target is None:
        return rescaled_image

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "polygons" in target:
        polygons = target["polygons"]
        scaled_ratio = torch.cat([torch.tensor([ratio_width, ratio_height])
                                 for _ in range(polygons.shape[1] // 2)], dim=0)
        scaled_polygons = polygons * scaled_ratio
        target["polygons"] = scaled_polygons

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        assert False
        # target['masks'] = interpolate(
        #     target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std, max_image_size=512):
        self.mean = mean
        self.std = std
        self.max_image_size = max_image_size

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        # h, w = image.shape[-2:]
        h, w = target["size"][0], target["size"][1]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / self.max_image_size
            target["boxes"] = boxes
        if "polygons" in target:
            polygons = target["polygons"]
            scale = torch.cat([torch.tensor([w, h], dtype=torch.float32)
                               for _ in range(polygons.shape[1] // 2)], dim=0)
            polygons = polygons / scale
            target["polygons"] = polygons
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img):
        if random.random() < self.prob:
            return self.tfm(img)
        else:
            return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, tartget=None):
        do_it = random.random() <= self.prob
        if not do_it:
            if tartget is not None:
                return img, tartget
            else:
                return img

        if tartget is not None:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            ), tartget
        else:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
