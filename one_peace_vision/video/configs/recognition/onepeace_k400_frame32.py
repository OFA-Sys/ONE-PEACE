_base_ = [
    '../_base_/models/onepeace.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(drop_path_rate=0.4, adapter_scale=0.5, num_frames=32),
    cls_head=dict(num_classes=400),
    test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/root/data/dataset/Kinetics/kinetics400/train'
data_root_val = '/root/data/dataset/Kinetics/kinetics400/academic_torrent/val_256'
ann_file_train = '/root/data/dataset/Kinetics/kinetics400/kinetics400_train_list_videos_fix.txt'
ann_file_val = '/root/data/dataset/Kinetics/kinetics400/k400_val_academic_torrent.txt'
ann_file_test = '/root/data/dataset/Kinetics/kinetics400/k400_val_academic_torrent.txt'
img_norm_cfg = dict(
    mean=[122.771, 116.746, 104.094], std=[68.5, 66.632, 70.323], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=4,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'cls_embedding': dict(decay_mult=0.),
                                                 'pos_embed': dict(decay_mult=0.),
                                                 'self_attn_layer_norm': dict(decay_mult=0.),
                                                 'final_layer_norm': dict(decay_mult=0.),
                                                 'image_layer_norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1), }))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=3
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/k400_onepeace_patch256.py'
find_unused_parameters = False

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
