# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import utils


@dataclass
class HingeLossConfig(FairseqDataclass):
    margin: float = 1.0
    num_choices: int = 4


@register_criterion("hinge_loss", dataclass=HingeLossConfig)
class HingeLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        margin=1.0,
        num_choices=4,
    ):
        super().__init__(task)
        self.margin = margin
        self.num_choices = num_choices

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        src_tokens = sample['net_input']['src_tokens']
        src_audios = sample['net_input']['src_audios'].repeat_interleave(self.num_choices, 0)
        audio_padding_masks = sample['net_input']['audio_padding_masks'].repeat_interleave(self.num_choices, 0)

        logits = model(
            src_tokens=src_tokens, src_audios=src_audios, audio_padding_masks=audio_padding_masks
        ).view(-1, self.num_choices)
        targets = sample['target']
        positive_logits = logits.gather(1, targets.unsqueeze(1))
        loss = torch.max(torch.tensor(0.0).to(logits.device), 1 + logits - positive_logits).sum()
        n_correct = logits.argmax(1).eq(targets).sum()

        sample_size = sample['nsentences']
        logging_output = {
            "loss": loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
            "n_correct": n_correct
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

        total = utils.item(sample_size)
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True