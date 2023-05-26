from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import CascadeROIHeads, FastRCNNConvFCHead

from models import OnePeace, get_onepeace_lr_decay_rate
from layers import FastRCNNOutputLayersSoftNms
from ..common.coco_loader_lsj_1280 import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model

train = model_zoo.get_config("common/train.py").train

model.pixel_mean = [122.7709383, 116.7460125, 104.09373615]
model.pixel_std = [68.5005327, 66.6321579, 70.32316305]
model.input_format = "RGB"
model.backbone = L(SimpleFeaturePyramid)(
    net=L(OnePeace)(
        attention_heads=24,
        bucket_size=80,
        pretrain_bucket_size=16,
        dropout=0.0,
        drop_path_rate=0.6,
        embed_dim=1536,
        ffn_embed_dim=6144,
        layers=40,
        out_feature="last_feat",
        rp_bias=False,
        use_decomposed_rel_pos=True,
        shared_rp_bias=True,
        use_checkpoint=True,
        window_size=16,
        window_block_indexes=(
            list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
            list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) +
            list(range(32, 35)) + list(range(36, 39))
        )
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1280,
)

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

model.roi_heads.mask_head.conv_norm = "LN"

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayersSoftNms)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.0,
            test_nms_thresh=0.6,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
            soft_nms_enabled=True,
            soft_nms_method="linear",
            soft_nms_sigma=0.5,
            soft_nms_prune=1e-3,
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

train.amp.enabled = True
train.ddp.fp16_compression = True
train.max_iter = 92187
train.checkpointer = dict(period=5000, max_to_keep=100)
train.eval_period = 5000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[81944, 88773],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_onepeace_lr_decay_rate, num_layers=40, lr_decay_rate=0.9)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
