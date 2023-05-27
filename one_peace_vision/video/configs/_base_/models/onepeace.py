# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='OnePeaceViT',
        attention_heads=24,
        bucket_size=16,
        num_frames=32,
        dropout=0.0,
        drop_path_rate=0.4,
        embed_dim=1536,
        ffn_embed_dim=6144,
        layers=40,
        rp_bias=False,
        shared_rp_bias=True,
        use_checkpoint=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=1536,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob'))
