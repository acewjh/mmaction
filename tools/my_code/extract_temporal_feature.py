import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers

import os.path as osp
import os
import numpy as np

def single_test(model, data_loader, temporal_feature_folder):
	def _save_result(result, action_dir, sample_name):
		output_path = osp.join(action_dir, temporal_feature_folder, '{}.npy'.format(sample_name))
		if not osp.exists(osp.dirname(output_path)):
			os.makedirs(osp.dirname(output_path))
		np.save(output_path, result)
	
	model.eval()
	results = []
	dataset = data_loader.dataset
	prog_bar = mmcv.ProgressBar(len(dataset))
	for i, data in enumerate(data_loader):
		with torch.no_grad():
			result = model(extract_temporal=True, **data)
		_save_result(np.squeeze(result), dataset.action_dir, data['sample_name'].data[0][0])
		batch_size = data['img_group_0'].data[0].size(0)
		for _ in range(batch_size):
			prog_bar.update()
	return results


def _data_func(data, device_id):
	data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
	return dict(return_loss=False, rescale=True, **data)


def parse_args():
	parser = argparse.ArgumentParser(description='Test an action recognizer')
	parser.add_argument('config', help='test config file path')
	parser.add_argument('checkpoint', help='checkpoint file')
	parser.add_argument(
		'--gpus', default=1, type=int, help='GPU number used for testing')
	parser.add_argument(
		'--proc_per_gpu',
		default=1,
		type=int,
		help='Number of processes per GPU')
	parser.add_argument('--temporal_feature_folder', type=str, default='flow_tsn_ucf101')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	
	cfg = mmcv.Config.fromfile(args.config)
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	cfg.data.test.test_mode = True
	
	if cfg.data.test.oversample == 'three_crop':
		cfg.model.spatial_temporal_module.spatial_size = 8
	
	dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
	##TODO: fix multi gpu
	if args.gpus == 1:
		model = build_recognizer(
			cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
		load_checkpoint(model, args.checkpoint, strict=True)
		model = MMDataParallel(model, device_ids=[0])
		
		# foo_input = torch.zeros((8, 3, 10, 224, 224)).cuda()
		# foo_output = model([1], None, img_group_0=foo_input, return_loss=False)
		
		data_loader = build_dataloader(
			dataset,
			imgs_per_gpu=1,
			workers_per_gpu=cfg.data.workers_per_gpu,
			num_gpus=1,
			dist=False,
			shuffle=False)
		single_test(model, data_loader, args.temporal_feature_folder)
	else:
		model_args = cfg.model.copy()
		model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
		model_type = getattr(recognizers, model_args.pop('type'))
		outputs = parallel_test(
			model_type,
			model_args,
			args.checkpoint,
			dataset,
			_data_func,
			range(args.gpus),
			workers_per_gpu=args.proc_per_gpu)


if __name__ == '__main__':
	main()
