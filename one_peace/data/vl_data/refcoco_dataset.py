# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import numpy as np
import torch

from ..base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
from ...utils import transforms as T


class RefCOCODataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_image_size=384
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        if self.split == 'train':
            self.transform = T.Compose([
                T.RandomResize([patch_image_size], max_size=patch_image_size),
                T.GaussianBlur(0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std, max_image_size=patch_image_size)
            ])
        else:
            self.transform = T.Compose([
                T.RandomResize([patch_image_size], max_size=patch_image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std, max_image_size=patch_image_size)
            ])

    def __getitem__(self, index):
        item = self.dataset[index]
        image, text, region_coord = item

        image = self.read_image(image)
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        text = self.process_text(text, self.max_src_length)
        src_item = self.encode_text(' {}'.format(text))

        example = {
            "id": index,
            "source_text": src_item,
            "source_image": patch_image,
            "target": patch_boxes["boxes"],
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return example
