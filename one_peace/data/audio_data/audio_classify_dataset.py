# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import soundfile as sf

from one_peace.data.base_dataset import BaseDataset


class AudioClassifyDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_duration=15,
        feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
        num_classes=200
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_duration = max_duration
        self.feature_encoder_spec = eval(feature_encoder_spec)
        self.num_classes = num_classes

    def __getitem__(self, index):
        uniq_id, audio, refs, duration = self.dataset[index]

        wav, curr_sample_rate = sf.read(audio, dtype="float32")
        feats = torch.tensor(wav)
        feats = self.audio_postprocess(feats, curr_sample_rate, self.max_duration)
        T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        audio_padding_mask = torch.zeros(T + 1).bool()

        label_item = torch.zeros(self.num_classes, dtype=torch.float)
        for label_id in refs.strip().split(','):
            label_item[int(label_id)] = 1
        label_item = label_item.unsqueeze(0)

        example = {
            "id": uniq_id,
            "source_audio": feats,
            "audio_padding_mask": audio_padding_mask,
            "target": label_item,
        }
        return example