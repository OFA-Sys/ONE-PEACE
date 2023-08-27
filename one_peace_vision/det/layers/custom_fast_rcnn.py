import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, Instances
from .soft_nms import batched_soft_nms

__all__ = ["fast_rcnn_inference_softnms", "FastRCNNOutputLayersSoftNms"]


logger = logging.getLogger(__name__)


def fast_rcnn_inference_softnms(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
):
    result_per_image = [
        fast_rcnn_inference_single_image_softnms(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh,
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune, topk_per_image, s_bf_per_img
        )
        for scores_per_image, boxes_per_image, image_shape, s_bf_per_img in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_softnms(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
        )
        scores[keep] = soft_nms_scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputLayersSoftNms(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        soft_nms_enabled: bool = False,
        soft_nms_method: str = "gaussian",
        soft_nms_sigma: float = 0.5,
        soft_nms_prune: float = 0.001,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
    ):
        super().__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
            use_fed_loss=use_fed_loss,
            use_sigmoid_ce=use_sigmoid_ce,
            get_fed_loss_cls_weights=get_fed_loss_cls_weights,
            fed_loss_num_classes=fed_loss_num_classes
        )
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_prune = soft_nms_prune

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference_softnms(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.soft_nms_enabled,
            self.soft_nms_method,
            self.soft_nms_sigma,
            self.soft_nms_prune,
            self.test_topk_per_image,
        )
