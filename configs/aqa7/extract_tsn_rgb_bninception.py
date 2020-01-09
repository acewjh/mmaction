# model settings
model = dict(
    type='TSN2D',
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
        type='RegHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.8,
        in_channels=1024,
        init_std=0.001,
        num_output=1,
        loss_func='ranking_mse',
        alpha=1.0,
        beta=100))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'AQA_SingleClassDataset'
img_norm_cfg = dict(
   mean=[104, 117, 128], std=[1, 1, 1], to_rgb=False)

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
	test=dict(
		type=dataset_type,
		action_dir='/home/wangjiahao/datasets/AQA-7/Actions/diving',
		subset='all',
		label_name='label',
		img_norm_cfg=img_norm_cfg,
		num_segments=103,
		new_length=1,
		new_step=1,
		random_shift=False,
		modality='RGB',
		image_tmpl='{:05d}.jpg',
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
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 100])
checkpoint_config = dict(interval=5)
workflow = [('train', 5), ('val', 1)]
# workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=5,
	ignore_last=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
metric_config = dict(
	metrics=['src', 'med']
)
# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/wangjiahao/experiments/AQA-7/diving/tsn_rgb_bninception_exp11'
load_from = './modelzoo/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth'
resume_from = None



