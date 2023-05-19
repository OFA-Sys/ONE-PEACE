# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from one_peace.data.base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD


class ImageTextRetrievalDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_image_size=256,
        image_augmentation='clip',
        min_scale=0.9
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        if image_augmentation == 'clip' and split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    patch_image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif image_augmentation == 'raw' and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        uniq_id, image, caption = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or uniq_id.isdigit() else uniq_id

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        if image is not None:
            image = Image.open(image).convert("RGB")
            patch_image = self.transform(image)
        else:
            patch_image = torch.randn((self.patch_image_size, self.patch_image_size))

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_image": patch_image
        }
        return example