# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import utils


@dataclass
class ClassifyCriterionConfig(FairseqDataclass):
    use_multi_label: bool = field(
        default=False, metadata={"help": "use multi label loss"}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("classify_criterion", dataclass=ClassifyCriterionConfig)
class ClassifyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        use_multi_label=False,
        label_smoothing=0.0
    ):
        super().__init__(task)
        self.use_multi_label = use_multi_label
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits = model(**sample['net_input'])
        targets = sample['target']
        if self.use_multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='sum')
            with torch.no_grad():
                n_correct = targets.gather(1, logits.argmax(1, keepdim=True)).sum()
        else:
            if targets.dim() == 2:
                log_probs = utils.log_softmax(logits, dim=-1)
                loss = (-targets * log_probs).sum()
                with torch.no_grad():
                    n_correct = (log_probs.exp() * targets).sum()
            else:
                loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction='sum')
                with torch.no_grad():
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