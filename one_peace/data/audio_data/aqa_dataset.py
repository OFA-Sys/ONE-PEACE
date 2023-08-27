# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch

from ..base_dataset import BaseDataset


class AQADataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_duration=15,
        feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_duration = max_duration
        self.feature_encoder_spec = eval(feature_encoder_spec)

    def __getitem__(self, index):
        item_tuple = self.dataset[index]
        uniq_id, audio, text, duration = item_tuple
        question, answer1, answer2, answer3, answer4, choice = text.strip().split('&&&')

        wav, curr_sample_rate = self.read_audio(audio)
        feats = torch.tensor(wav)
        feats = self.audio_postprocess(feats, curr_sample_rate, self.max_duration)
        T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        audio_padding_mask = torch.zeros(T + 1).bool()

        label_item = torch.LongTensor([int(choice)])
        src_item = []
        src_item += [self.encode_text(' {} answer: {}'.format(question, answer1))]
        src_item += [self.encode_text(' {} answer: {}'.format(question, answer2))]
        src_item += [self.encode_text(' {} answer: {}'.format(question, answer3))]
        src_item += [self.encode_text(' {} answer: {}'.format(question, answer4))]

        example = {
            "id": index,
            "source_text": src_item,
            "source_audio": feats,
            "audio_padding_mask": audio_padding_mask,
            "target": label_item
        }
        return example