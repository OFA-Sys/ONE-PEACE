# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD


class VqaDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_image_size=480,
        answer_cnt=3129
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size
        self.answer_cnt = answer_cnt

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __getitem__(self, index):
        uniq_id, image, question, refs = self.dataset[index]
        uniq_id = int(uniq_id)

        image = self.read_image(image)
        patch_image = self.transform(image)

        question = self.process_text(question)
        src_item = self.encode_text(' {}'.format(question), self.max_src_length)

        label_item = torch.zeros(self.answer_cnt, dtype=torch.float)
        for item in refs.strip().split('&&'):
            label, label_id, conf = item.split('|!+')
            label_item[int(label_id)] = float(conf)
        label_item = label_item.unsqueeze(0)

        example = {
            "id": uniq_id,
            "source_text": src_item,
            "source_image": patch_image,
            "target": label_item
        }
        return example
