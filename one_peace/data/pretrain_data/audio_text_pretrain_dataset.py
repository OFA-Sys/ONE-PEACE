# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math

import torch

from ..base_dataset import BaseDataset
from ...utils.data_utils import get_whole_word_mask, compute_block_mask_1d


class AudioTextPretrainDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        max_duration=15,
        feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
        audio_mask_ratio=0.55,
        al_text_mask_ratio=0.4,
        al_audio_mask_ratio=0.45,
        audio_mask_prob_adjust=0.1,
        audio_mask_length=5
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.max_duration = max_duration
        self.feature_encoder_spec = eval(feature_encoder_spec)
        self.mask_whole_word = (get_whole_word_mask(bpe, dictionary))

        self.audio_mask_ratio = audio_mask_ratio
        self.al_text_mask_ratio = al_text_mask_ratio
        self.al_audio_mask_ratio = al_audio_mask_ratio

        self.audio_mask_prob_adjust = audio_mask_prob_adjust
        self.audio_mask_length = audio_mask_length

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        uniq_id, audio, caption, duration = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or uniq_id.isdigit() else uniq_id

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length, append_eos=False)
        al_text_mask_indices = self.add_whole_word_mask(text_src_item, self.al_text_mask_ratio)
        text_src_item = torch.cat([text_src_item, torch.LongTensor([self.eos])])

        if audio is not None:
            wav, curr_sample_rate = self.read_audio(audio)
            feats = torch.tensor(wav)
        else:
            feats = torch.randn(16000)
            curr_sample_rate = 16000
        feats = self.audio_postprocess(feats, curr_sample_rate, self.max_duration)

        T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        audio_mask_indices = compute_block_mask_1d(
            shape=(1, T),
            mask_prob=self.audio_mask_ratio,
            mask_length=self.audio_mask_length,
            mask_prob_adjust=self.audio_mask_prob_adjust
        ).squeeze(0).bool()
        al_audio_mask_indices = compute_block_mask_1d(
            shape=(1, T),
            mask_prob=self.al_audio_mask_ratio,
            mask_length=self.audio_mask_length,
            mask_prob_adjust=self.audio_mask_prob_adjust
        ).squeeze(0).bool()
        audio_padding_mask = torch.zeros(T+1).bool()

        audio_mask_indices = torch.cat([torch.BoolTensor([False]), audio_mask_indices])
        audio_preserve_ids = (~audio_mask_indices).nonzero(as_tuple=True)[0]
        al_text_mask_indices = torch.cat([torch.BoolTensor([False]), al_text_mask_indices, torch.BoolTensor([False])])
        al_text_preserve_ids = (~al_text_mask_indices).nonzero(as_tuple=True)[0]
        al_audio_mask_indices = torch.cat([torch.BoolTensor([False]), al_audio_mask_indices])
        al_audio_preserve_ids = (~al_audio_mask_indices).nonzero(as_tuple=True)[0]

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_audio": feats,
            "audio_padding_mask": audio_padding_mask,
            "audio_mask_indices": audio_mask_indices,
            "audio_preserve_ids": audio_preserve_ids,
            "al_text_mask_indices": al_text_mask_indices,
            "al_text_preserve_ids": al_text_preserve_ids,
            "al_audio_mask_indices": al_audio_mask_indices,
            "al_audio_preserve_ids": al_audio_preserve_ids,
        }
        return example

    def add_whole_word_mask(self, source, p):
        is_word_start = self.mask_whole_word.gather(0, source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        assert num_to_mask != 0

        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_indices = torch.zeros(len(source)).bool()
        mask_indices[indices] = True

        is_word_start = torch.cat([is_word_start, torch.Tensor([255]).type_as(is_word_start)])
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_indices[indices] = True

        return mask_indices