# model settings
model = dict(
    type='TSN2D',
    modality='Flow',
    in_channels=10,
    backbone=dict(
        type='BNInception',
        pretrained='open-mmlab://bninception_caffe',
        bn_eval=False,
        partial_bn=True),
    spatial_temporal_module=dict(
        type='SimpleSpatialModule',
        spatial_type='avg',
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.7,
        in_channels=1024,
        num_classes=101))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'AQA_SingleClassDataset'
img_norm_cfg = dict(
    mean=[128], std=[1], to_rgb=False)
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        action_dir='/mnt/sdb1/wjh/datasets/AQA-7/Actions/trampoline',
        subset='all',
        label_name='label',
        img_norm_cfg=img_norm_cfg,
        num_segments=20,
        new_length=5,
        new_step=1,
        random_shift=False,
        modality='Flow',
        image_tmpl='flow_{}_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[190, 300])
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 340
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_2d_flow_bninception_seg_3_f1s1_b32_g8_lr_0.005'
load_from = None
resume_from = None



