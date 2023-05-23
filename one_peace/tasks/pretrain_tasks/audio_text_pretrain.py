# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Optional
import logging
import json
import torch

from fairseq.tasks import register_task
from fairseq.utils import move_to_cuda

from tasks.base_task import BaseTask, BaseTaskConfig
from data.pretrain_data.audio_text_pretrain_dataset import AudioTextPretrainDataset
from metrics import Recall

logger = logging.getLogger(__name__)


@dataclass
class AudioTextPretrainConfig(BaseTaskConfig):
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "validation file, json format."},
    )

    audio_mask_ratio: float = field(
        default=0.55,
        metadata={"help": "mask ratio of audio data"}
    )
    al_text_mask_ratio: float = field(
        default=0.4,
        metadata={"help": "text mask ratio of audio-language data"}
    )
    al_audio_mask_ratio: float = field(
        default=0.45,
        metadata={"help": "audio mask ratio of audio-language data"}
    )

    audio_mask_prob_adjust: float = 0.1
    audio_mask_length: int = 5


@register_task("audio_text_pretrain", dataclass=AudioTextPretrainConfig)
class AudioTextPretrainTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Recall()
        self.text_ids = None
        self.texts = None

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        if self.text_ids is None and self.cfg.valid_file is not None:
            self.text_ids = []
            self.texts = []
            for text_id, text_list in json.load(open(self.cfg.valid_file)).items():
                for text in text_list:
                    self.text_ids.append(int(text_id))
                    self.texts.append(text)
            self.text_ids = torch.tensor(self.text_ids).cuda()

        self.datasets[split] = AudioTextPretrainDataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            max_duration=self.cfg.max_duration,
            feature_encoder_spec=self.cfg.feature_encoder_spec,
            audio_mask_ratio=self.cfg.audio_mask_ratio,
            al_text_mask_ratio=self.cfg.al_text_mask_ratio,
            al_audio_mask_ratio=self.cfg.al_audio_mask_ratio,
            audio_mask_prob_adjust=self.cfg.audio_mask_prob_adjust,
            audio_mask_length=self.cfg.audio_mask_length,
        )

    @torch.no_grad()
    def begin_valid_epoch(self, epoch, model, subset):
        assert self.text_ids is not None and self.texts is not None
        model.eval()

        dataset = self.datasets[subset]

        text_logits_list = []
        samples_list = []
        for text in self.texts:
            text = "This is a sound of " + text
            item_tuple = (0, None, text, 1, None)
            sample = dataset.__getitem__(0, item_tuple)
            samples_list.append(sample)
        samples = dataset.collater(samples_list)
        samples = move_to_cuda(samples)
        src_tokens = samples["net_input"]["src_tokens"]
        text_logits, _ = model(src_tokens=src_tokens, encoder_type='text')
        text_logits_list.append(text_logits)

        text_logits = torch.cat(text_logits_list, dim=0)
        self.metric.initialize(self.text_ids, text_logits)

    @torch.no_grad()
    def valid_step(self, sample, model, criterion, is_dummy_batch):
        loss = 0
        sample_size = len(sample['id'])
        logging_output = {'nsentences': 1, 'ntokens': 1}
        if not is_dummy_batch:
            model.eval()
            self.eval_step(model, sample)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def eval_step(self, model, sample):
        src_audios = sample["net_input"]["src_audios"]
        audio_padding_masks = sample["net_input"]["audio_padding_masks"]
        audio_ids = torch.tensor(sample['id']).to(src_audios.device)
        audio_logits, _ = model(src_audios=src_audios, audio_padding_masks=audio_padding_masks, encoder_type='audio')
        self.metric.compute(audio_ids, audio_logits)

    @torch.no_grad()
    def merge_results(self, output_predict=False):
        stats =  self.metric.merge_results(output_predict=output_predict)
        for key in list(stats.keys()):
            if key.startswith('img'):
                stats[key.replace('img', 'audio')] = stats[key]
                del stats[key]
        return stats