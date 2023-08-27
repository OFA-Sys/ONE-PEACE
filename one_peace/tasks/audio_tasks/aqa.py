# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import torch

from fairseq.tasks import register_task

from ..base_task import BaseTask, BaseTaskConfig
from ...data.audio_data.aqa_dataset import AQADataset
from ...metrics import Accuracy

logger = logging.getLogger(__name__)


@dataclass
class AQAConfig(BaseTaskConfig):
    pass


@register_task("aqa", dataclass=AQAConfig)
class AQATask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Accuracy()

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        self.datasets[split] = AQADataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_duration=self.cfg.max_duration,
            feature_encoder_spec=self.cfg.feature_encoder_spec
        )

    @torch.no_grad()
    def eval_step(self, model, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_audios = sample['net_input']['src_audios'].repeat_interleave(4, 0)
        audio_padding_masks = sample['net_input']['audio_padding_masks'].repeat_interleave(4, 0)
        logits = model(
            src_tokens=src_tokens, src_audios=src_audios, audio_padding_masks=audio_padding_masks
        ).view(-1, 4)
        ids = torch.tensor(sample['id']).to(logits.device)
        self.metric.compute(ids, logits, sample['target'])
