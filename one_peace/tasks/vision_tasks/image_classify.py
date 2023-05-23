# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional
from dataclasses import dataclass, field
import logging
import torch

from fairseq.tasks import register_task

from tasks.base_task import BaseTask, BaseTaskConfig
from data.vision_data.image_classify_dataset import ImageClassifyDataset
from metrics import Accuracy

logger = logging.getLogger(__name__)


@dataclass
class ImageClassifyConfig(BaseTaskConfig):
    color_jitter: float = field(
        default=0.4,
        metadata={"help": ""}
    )
    center_crop: bool = field(
        default=False,
        metadata={"help": ""}
    )
    raw_transform: bool = field(
        default=False,
        metadata={"help": ""}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    mixup: float = field(
        default=0.0,
        metadata={"help": "mixup alpha, mixup enabled if > 0."},
    )
    cutmix: float = field(
        default=0.0,
        metadata={"help": "cutmix alpha, cutmix enabled if > 0."},
    )
    cutmix_minmax: Optional[str] = field(
        default=None,
        metadata={"help": "cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)"},
    )
    mixup_prob: float = field(
        default=1.0,
        metadata={"help": "Probability of performing mixup or cutmix when either/both is enabled"},
    )
    mixup_switch_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of switching to cutmix when both mixup and cutmix enabled"},
    )
    mixup_mode: str = field(
        default='batch',
        metadata={"help": 'How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'},
    )


@register_task("image_classify", dataclass=ImageClassifyConfig)
class ImageClassifyTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Accuracy()

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        self.datasets[split] = ImageClassifyDataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            patch_image_size=self.cfg.patch_image_size,
            color_jitter=self.cfg.color_jitter,
            center_crop=self.cfg.center_crop,
            raw_transform=self.cfg.raw_transform,
            mixup=self.cfg.mixup,
            cutmix=self.cfg.cutmix,
            cutmix_minmax=self.cfg.cutmix_minmax,
            mixup_prob=self.cfg.mixup_prob,
            mixup_switch_prob=self.cfg.mixup_switch_prob,
            mixup_mode=self.cfg.mixup_mode,
            num_classes=self.cfg.num_classes,
            label_smoothing=self.cfg.label_smoothing
        )

    @torch.no_grad()
    def eval_step(self, model, sample):
        logits = model(**sample['net_input'])
        ids = torch.tensor(sample['id']).to(logits.device)
        self.metric.compute(ids, logits, sample['target'])
