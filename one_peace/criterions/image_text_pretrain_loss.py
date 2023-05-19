# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


def adjust_label_smoothed_nll_loss(lprobs, target, epsilon=0.0):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if epsilon != 0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    else:
        loss = nll_loss
    return loss.mean()


@torch.no_grad()
def gather_without_grad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@dataclass
class ImageTextPretrainLossConfig(FairseqDataclass):
    dcl_text_alpha: float=0.5
    dcl_image_alpha: float=1.0
    dcl_vl_text_alpha: float=0.5
    dcl_vl_image_alpha: float=0.5
    dcl_logit_scale: float=2.5
    itc_without_ls: bool=True

    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("image_text_pretrain_loss", dataclass=ImageTextPretrainLossConfig)
class ImageTextPretrainLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        dcl_text_alpha,
        dcl_image_alpha,
        dcl_vl_text_alpha,
        dcl_vl_image_alpha,
        dcl_logit_scale,
        itc_without_ls,
        label_smoothing=0.0
    ):
        super().__init__(task)
        self.dcl_text_alpha = dcl_text_alpha
        self.dcl_image_alpha = dcl_image_alpha
        self.dcl_vl_text_alpha = dcl_vl_text_alpha
        self.dcl_vl_image_alpha = dcl_vl_image_alpha
        self.dcl_logit_scale = dcl_logit_scale
        self.itc_without_ls = itc_without_ls
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        src_tokens = sample['net_input']['src_tokens']
        text_preserve_ids = sample['net_input']['text_preserve_ids']
        text_mask_indices = sample['net_input']['text_mask_indices']

        src_images = sample['net_input']['src_images']
        image_preserve_ids = sample['net_input']['image_preserve_ids']
        image_mask_indices = sample['net_input']['image_mask_indices']

        vl_text_preserve_ids = sample['net_input']['vl_text_preserve_ids']
        vl_text_mask_indices = sample['net_input']['vl_text_mask_indices']
        vl_image_preserve_ids = sample['net_input']['vl_image_preserve_ids']
        vl_image_mask_indices = sample['net_input']['vl_image_mask_indices']

        text_logits, teacher_text_features = model(src_tokens=src_tokens, encoder_type='text')
        image_logits, teacher_image_features = model(src_images=src_images, encoder_type='image')
        text_logits_all = gather_without_grad(text_logits) if dist.is_initialized() else text_logits.data
        image_logits_all = gather_without_grad(image_logits) if dist.is_initialized() else image_logits.data
        with torch.no_grad():
            teacher_vl_text_features, teacher_vl_image_features = model(
                src_tokens=src_tokens, src_images=src_images, encoder_type='vl'
            )

        student_text_features, _, _ = model(
            src_tokens=src_tokens,
            text_preserve_ids=text_preserve_ids,
            encoder_type='text'
        )
        _, student_image_features, _ = model(
            src_images=src_images,
            image_preserve_ids=image_preserve_ids,
            encoder_type='image'
        )
        student_vl_text_featrues, student_vl_image_features, _ = model(
            src_tokens=src_tokens, text_preserve_ids=vl_text_preserve_ids,
            src_images=src_images, image_preserve_ids=vl_image_preserve_ids,
            encoder_type='vl'
        )

        logit_scale_exp = model(return_logit_scale=True)

        padding_masks = src_tokens.eq(1)
        dcl_text_loss = self.compute_dcl_loss(
            student_text_features, teacher_text_features, text_mask_indices, padding_masks=padding_masks
        )
        dcl_image_loss = self.compute_dcl_loss(
            student_image_features, teacher_image_features, image_mask_indices,
        )
        dcl_vl_text_loss = self.compute_dcl_loss(
            student_vl_text_featrues, teacher_vl_text_features, vl_text_mask_indices, padding_masks=padding_masks
        )
        dcl_vl_image_loss = self.compute_dcl_loss(
            student_vl_image_features, teacher_vl_image_features, vl_image_mask_indices,
        )

        itc_loss, i2t_ncorrect, t2i_ncorrect = self.compute_itc_loss(
            image_logits, text_logits,
            image_logits_all, text_logits_all,
            logit_scale_exp
        )

        loss = itc_loss + \
               self.dcl_text_alpha * dcl_text_loss + self.dcl_image_alpha * dcl_image_loss + \
               self.dcl_vl_text_alpha * dcl_vl_text_loss + self.dcl_vl_image_alpha * dcl_vl_image_loss
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "itc_loss": itc_loss.data,
            "dcl_text_loss": dcl_text_loss.data,
            "dcl_image_loss": dcl_image_loss.data,
            "dcl_vl_text_loss": dcl_vl_text_loss.data,
            "dcl_vl_image_loss": dcl_vl_image_loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
            "i2t_ncorrect": i2t_ncorrect,
            "t2i_ncorrect": t2i_ncorrect,
            "logit_scale_exp": logit_scale_exp
        }
        return loss, sample_size, logging_output

    def compute_itc_loss(self, image_logits, text_logits, image_logits_all, text_logits_all, logit_scale_exp):
        slice_id = dist.get_rank() if dist.is_initialized() else 0
        bsz = image_logits.size(0)
        start_idx = bsz * slice_id
        end_idx = start_idx + bsz
        targets = torch.arange(start_idx, end_idx).to(image_logits.device)

        sim_i2t = logit_scale_exp * image_logits @ text_logits_all.t()
        sim_t2i = logit_scale_exp * text_logits @ image_logits_all.t()
        log_sim_i2t = utils.log_softmax(sim_i2t, dim=-1).type_as(sim_i2t)
        log_sim_t2i = utils.log_softmax(sim_t2i, dim=-1).type_as(sim_t2i)
        if self.itc_without_ls:
            i2t_loss = adjust_label_smoothed_nll_loss(log_sim_i2t, targets)
            t2i_loss = adjust_label_smoothed_nll_loss(log_sim_t2i, targets)
        else:
            i2t_loss = adjust_label_smoothed_nll_loss(log_sim_i2t, targets, self.label_smoothing)
            t2i_loss = adjust_label_smoothed_nll_loss(log_sim_t2i, targets, self.label_smoothing)
        itc_loss = (i2t_loss + t2i_loss) / 2

        with torch.no_grad():
            i2t_preds = sim_i2t.argmax(dim=1)
            t2i_preds = sim_t2i.argmax(dim=1)
            i2t_ncorrect = (i2t_preds == targets).float().sum()
            t2i_ncorrect = (t2i_preds == targets).float().sum()

        return itc_loss, i2t_ncorrect, t2i_ncorrect

    def compute_dcl_loss(self, student_features, teacher_features, mask_indices, padding_masks=None):
        embed_dim = student_features.size(-1)
        teacher_features = teacher_features.detach()
        student_out = student_features[:, 1:, :].reshape(-1, embed_dim)
        teacher_out = teacher_features[:, 1:, :].reshape(-1, embed_dim)
        mask_indices = mask_indices[:, 1:].flatten()
        if padding_masks is not None:
            non_padding_mask_indices = torch.nonzero((~padding_masks).flatten(), as_tuple=False).flatten()
            student_out = student_out[non_padding_mask_indices]
            teacher_out = teacher_out[non_padding_mask_indices]
            mask_indices = mask_indices[non_padding_mask_indices]

        indices = torch.nonzero(mask_indices, as_tuple=False).flatten()
        targets = torch.arange(student_out.size(0)).to(student_out.device)[indices]
        orig_type = student_out.dtype
        mask_student_out = F.normalize(student_out[indices].float(), dim=1).to(orig_type)
        teacher_out = F.normalize(teacher_out.float(), dim=1).to(orig_type)

        sim_stu2tea = self.mask_logit_scale * mask_student_out @ teacher_out.t()
        log_sim_stu2tea = utils.log_softmax(sim_stu2tea, dim=-1).type_as(sim_stu2tea)
        loss = adjust_label_smoothed_nll_loss(log_sim_stu2tea, targets, self.label_smoothing)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        itc_loss_sum = sum(log.get("itc_loss", 0) for log in logging_outputs)
        dcl_text_loss_sum = sum(log.get("dcl_text_loss", 0) for log in logging_outputs)
        dcl_image_loss_sum = sum(log.get("dcl_image_loss", 0) for log in logging_outputs)
        dcl_vl_text_loss_sum = sum(log.get("dcl_vl_text_loss", 0) for log in logging_outputs)
        dcl_vl_image_loss_sum = sum(log.get("dcl_vl_image_loss", 0) for log in logging_outputs)
        logit_scale_exp_sum = sum(log.get("logit_scale_exp", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "itc_loss", itc_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_text_loss", dcl_text_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_image_loss", dcl_image_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_vl_text_loss", dcl_vl_text_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_vl_image_loss", dcl_vl_image_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "logit_scale_exp", logit_scale_exp_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

        if len(logging_outputs) > 0 and "i2t_ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("i2t_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "i2t_accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "t2i_ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("t2i_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "t2i_accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True