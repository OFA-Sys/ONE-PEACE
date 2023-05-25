# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform
from timm.data.mixup import Mixup

from .. import collate_fn
from ..base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
from ...utils.randaugment import RandomAugment
from ...utils import transforms as utils_transforms


class ImageClassifyDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=40,
        patch_image_size=384,
        color_jitter=0.4,
        center_crop=False,
        raw_transform=False,
        mixup=0.0,
        cutmix=0.0,
        cutmix_minmax=None,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mixup_mode='batch',
        num_classes=0,
        label_smoothing=0.0
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        self.mixup_fn = None
        if mixup > 0 or cutmix > 0. or cutmix_minmax is not None:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                label_smoothing=label_smoothing, num_classes=num_classes)

        if self.split == 'train':
            if raw_transform:
                self.transform = transforms.Compose([
                    transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
                    utils_transforms.RandomDistortion(0.4, 0.4, 0.4, 0, 0.5),
                    utils_transforms.GaussianBlur(0.5),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(2, 7, isPIL=True, augs=['Identity', 'Equalize', 'Brightness', 'Sharpness',
                                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                self.transform = create_transform(
                    input_size=patch_image_size,
                    is_training=True,
                    color_jitter=color_jitter,
                    auto_augment='rand-m9-mstd0.5-inc1',
                    interpolation='bicubic',
                    re_prob=0.25,
                    re_mode='pixel',
                    re_count=1,
                    mean=mean,
                    std=std,
                )
        elif center_crop:
            self.transform = transforms.Compose([
                transforms.Resize(patch_image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(patch_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([patch_image_size, patch_image_size], interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __getitem__(self, index):
        image, label = self.dataset[index]
        assert 0 < int(label) <= 1000
        label_item = torch.LongTensor([int(label) - 1])

        image = self.read_image(image)
        patch_image = self.transform(image)

        example = {
            "id": index,
            "source_image": patch_image,
            "target": label_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data required for the task
        """
        if len(samples) == 0:
            return {}

        batch = collate_fn(samples, pad_idx=self.pad)
        if self.split == 'train' and self.mixup_fn is not None:
            src_tokens = batch['net_input']['src_tokens']
            src_images = batch['net_input']['src_images']
            targets = batch['target']
            if len(src_images) % 2 != 0:
                src_images = torch.cat([src_images, src_images[:1]], dim=0)
                targets = torch.cat([targets, targets[:1]], dim=0)
                if src_tokens is not None:
                    src_tokens = torch.cat([src_tokens, src_tokens[:1]], dim=0)
            src_images, targets = self.mixup_fn(src_images, targets)
            batch['net_input']['src_tokens'] = src_tokens
            batch['net_input']['src_images'] = src_images
            batch['target'] = targets

        return batch