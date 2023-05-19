# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import torch

from fairseq.tasks import register_task

from one_peace.tasks.base_task import BaseTask, BaseTaskConfig
from one_peace.data.vl_data.refcoco_dataset import RefCOCODataset
from one_peace.metrics import IouAcc

logger = logging.getLogger(__name__)


@dataclass
class RefCOCOConfig(BaseTaskConfig):
    pass


@register_task("refcoco", dataclass=RefCOCOConfig)
class RefCOCOTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = IouAcc()

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        self.datasets[split] = RefCOCODataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            patch_image_size=self.cfg.patch_image_size,
        )

    @torch.no_grad()
    def eval_step(self, model, sample):
        output_coords = model(**sample['net_input']).sigmoid()
        hyps = output_coords * self.cfg.patch_image_size
        hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
        ids = torch.tensor(sample['id']).to(hyps.device)
        self.metric.compute(ids, hyps, sample['region_coords'])
