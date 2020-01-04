from __future__ import division

import argparse
from mmcv import Config
from mmcv.runner import obj_from_dict

from mmaction import __version__
from mmaction import datasets
from mmaction.datasets import get_trimmed_dataset
from mmaction.apis import (init_dist, get_root_logger,
						   set_random_seed)
from mmaction.apis.my_train import train_network
from mmaction.models import build_recognizer
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

def parse_args():
	parser = argparse.ArgumentParser(description='Train an action recognizer')
	parser.add_argument('config', help='train config file path')
	parser.add_argument('--work_dir', help='the dir to save logs and models')
	parser.add_argument(
		'--resume_from', help='the checkpoint file to resume from')
	parser.add_argument(
		'--validate',
		action='store_true',
		help='whether to evaluate the checkpoint during training')
	parser.add_argument(
		'--gpus',
		type=int,
		default=1,
		help='number of gpus to use '
		'(only applicable to non-distributed training)')
	parser.add_argument('--seed', type=int, default=None, help='random seed')
	parser.add_argument(
		'--launcher',
		choices=['none', 'pytorch', 'slurm', 'mpi'],
		default='none',
		help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)
	args = parser.parse_args()

	return args


def main():
	args = parse_args()

	cfg = Config.fromfile(args.config)
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	# update configs according to CLI args
	if args.work_dir is not None:
		cfg.work_dir = args.work_dir
	if not os.path.exists(cfg.work_dir):
		os.makedirs(cfg.work_dir)
	cmd = 'cp {} {}/'.format(args.config, cfg.work_dir)
	os.system(cmd)
	if args.resume_from is not None:
		cfg.resume_from = args.resume_from
	cfg.gpus = args.gpus
	if cfg.checkpoint_config is not None:
		# save mmaction version in checkpoints as meta data
		cfg.checkpoint_config.meta = dict(
			mmact_version=__version__, config=cfg.text)

	# init distributed env first, since logger depends on the dist info.
	if args.launcher == 'none':
		distributed = False
	else:
		distributed = True
		init_dist(args.launcher, **cfg.dist_params)

	# init logger before other steps
	logger = get_root_logger(cfg.log_level)
	logger.info('Distributed training: {}'.format(distributed))

	# set random seeds
	if args.seed is not None:
		logger.info('Set random seed to {}'.format(args.seed))
		set_random_seed(args.seed)

	model = build_recognizer(
		cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
	
	dataset_list = []
	train_dataset = obj_from_dict(cfg.data.train, datasets)
	dataset_list.append(train_dataset)
	if len(cfg.workflow) > 1 and args.validate:
		val_dataset = obj_from_dict(cfg.data.val, datasets)
		dataset_list.append(val_dataset)
	train_network(
		model,
		dataset_list,
		cfg,
		distributed=distributed,
		validate=args.validate,
		logger=logger)


if __name__ == '__main__':
	main()
