# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from one_peace.data.base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
from one_peace.utils.randaugment import RandomAugment
import one_peace.utils.transforms as utils_transforms


class Nlvr2Dataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=40,
        patch_image_size=384
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        if split == 'train':
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
            self.transform = transforms.Compose([
                transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __getitem__(self, index):
        uniq_id, text, image1, image2, label = self.dataset[index]
        if label == 'True':
            label_item = torch.LongTensor([0])
        elif label == 'False':
            label_item = torch.LongTensor([1])
        else:
            raise NotImplementedError

        image1 = Image.open(image1).convert("RGB")
        image2 = Image.open(image2).convert("RGB")
        patch_image1 = self.transform(image1)
        patch_image2 = self.transform(image2)

        text = self.process_text(text)
        src_item = self.encode_text(' {}'.format(text), self.max_src_length)

        example = {
            "id": index,
            "source_text": src_item,
            "source_image": patch_image1,
            "source_image2": patch_image2,
            "target": label_item
        }
        return example
