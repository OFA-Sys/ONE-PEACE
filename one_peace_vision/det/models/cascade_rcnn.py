from typing import List
import torch
from torch import nn

from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.roi_heads import CascadeROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher


class CustomCascadeROIHeads(CascadeROIHeads):
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        maskness_thresh: float = None,
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )
        self.maskness_thresh = maskness_thresh

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # optionally update score using maskness
            if self.maskness_thresh is not None:
                for pred_inst in pred_instances:
                    # pred_inst._fields.keys():  dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks'])
                    pred_masks = pred_inst.pred_masks  # (num_inst, 1, 28, 28)
                    scores = pred_inst.scores  # (num_inst, )
                    # sigmoid already applied
                    binary_masks = pred_masks > self.maskness_thresh
                    seg_scores = (pred_masks * binary_masks.float()).sum((1, 2, 3)) / binary_masks.sum((1, 2, 3))
                    seg_scores[binary_masks.sum((1, 2, 3)) == 0] = 0  # avoid nan
                    updated_scores = scores * seg_scores
                    pred_inst.set('scores', updated_scores)
                    # update order
                    scores, indices = updated_scores.sort(descending=True)
                    pred_inst = pred_inst[indices]
                    assert (pred_inst.scores == scores).all()
            return pred_instances, {}

