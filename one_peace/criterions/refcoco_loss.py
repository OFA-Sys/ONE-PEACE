# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class RefCOCOCriterionConfig(FairseqDataclass):
    pass


@register_criterion("refcoco_criterion", dataclass=RefCOCOCriterionConfig)
class RefCOCOCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        output_coords = model(**sample['net_input']).sigmoid()
        targets = sample['target']
        loss = F.l1_loss(output_coords, targets, reduction='sum')
        loss = loss / sample['nsentences']

        valid_indices = (output_coords[:, :2] < output_coords[:, 2:]).all(1)
        valid_output_coord = output_coords[valid_indices]
        valid_targets = targets[valid_indices]
        ious = torch.diag(generalized_box_iou(valid_output_coord, valid_targets))
        loss_ious = (1 - ious).mean()
        loss += loss_ious

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True