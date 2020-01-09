import os
import numpy as np
import os.path as osp
import torch
from collections import OrderedDict
from mmcv.runner.hooks import Hook
from mmcv.runner.utils import obj_from_dict
from .metrics import accuracy, src, med
from torch.nn.utils import clip_grad

class CacheOutputHook(Hook):
	def after_train_iter(self, runner):
		if not hasattr(runner, 'output_cache'):
			runner.output_cache = OrderedDict()
		for key, value in runner.outputs['output'].items():
			if not key in runner.output_cache.keys():
				runner.output_cache[key] = np.squeeze(value.cpu().detach().numpy())
			else:
				runner.output_cache[key] = np.hstack((runner.output_cache[key], np.squeeze(value.cpu().detach().numpy())))
	
	def before_train_epoch(self, runner):
		runner.output_cache = OrderedDict()
	
	def after_val_iter(self, runner):
		self.after_train_iter(runner)
	
	def before_val_epoch(self, runner):
		self.before_train_epoch(runner)
		
class CalMetricsHook(Hook):
	def __init__(self, metrics):
		for mtr in metrics:
			assert mtr in ['accuracy', 'src', 'med']
		self.metrics = list(set(metrics))
		self.metrics_funcs = []
		for mtr in self.metrics:
			if mtr == 'accuracy':
				self.metrics_funcs.append(accuracy)
			elif mtr == 'src':
				self.metrics_funcs.append(src)
			else:
				self.metrics_funcs.append(med)
			
	def after_train_epoch(self, runner):
		for i, mtr in enumerate(self.metrics):
			performance = self.metrics_funcs[i](runner.output_cache['output'], runner.output_cache['labels'])
			runner.log_buffer.update({mtr:performance})
		runner.log_buffer.average()
		
	def after_val_epoch(self, runner):
		self.after_train_epoch(runner)
		
class SaveOutputHook(Hook):
	def __init__(self):
		self.val_label_flag = True
	
	def after_train_epoch(self, runner):
		self.save_output('train', runner, 'pred_epoch_{}.npy', runner.output_cache['output'])
		self.save_output('train', runner, 'labels_epoch_{}.npy', runner.output_cache['labels'])
		
	def after_val_epoch(self, runner):
		self.save_output('val', runner, 'pred_epoch_{}.npy', runner.output_cache['output'])
		if self.val_label_flag:
			self.val_label_flag = False
			self.save_output('val', runner, 'labels_epoch_{}.npy', runner.output_cache['labels'])
		
	def save_output(self, mode, runner, tmpl, tensor):
		epoch = runner.epoch + 1 if mode == 'train' else runner.epoch
		save_path = osp.join(runner.work_dir, 'output', mode, tmpl.format(epoch))
		if not osp.exists(osp.dirname(save_path)):
			os.makedirs(osp.dirname(save_path))
		np.save(save_path, tensor)

class RenormalizeLossHook(Hook):
	def after_train_iter(self, runner):
		runner.model._modules['module']._modules['cls_head']._modules['loss_func'].renormalize()

class GradNormOptimizerHook(Hook):

	def __init__(self, gn_optim_config, grad_clip=None, gnpmt_start_idx=-4, gnpmt_end_idx=-2):
		self.grad_clip = grad_clip
		self.gn_optimizer = None
		self.gnpmt_start_idx = gnpmt_start_idx
		self.gnpmt_end_idx = gnpmt_end_idx
		self.gn_optim_config = gn_optim_config

	def clip_grads(self, params):
		clip_grad.clip_grad_norm_(
			filter(lambda p: p.requires_grad, params), **self.grad_clip)

	def after_train_iter(self, runner):
		if self.gn_optimizer is None:
			self.gn_optimizer = obj_from_dict(self.gn_optim_config, torch.optim,
                                      dict(params=runner.optimizer.param_groups[0]['params']
									  [self.gnpmt_start_idx:self.gnpmt_end_idx]))
			del runner.optimizer.param_groups[0]['params'][self.gnpmt_start_idx:self.gnpmt_end_idx]
		self.gn_optimizer.zero_grad()
		runner.outputs['loss_grad'].backward(retain_graph=True)
		self.gn_optimizer.step()
		runner.log_buffer.update(dict(loss_weight_0=self.gn_optimizer.param_groups[0]['params'][0].item(),
									  loss_weight_1=self.gn_optimizer.param_groups[0]['params'][1].item()))

		runner.optimizer.zero_grad()
		runner.outputs['loss'].backward()
		if self.grad_clip is not None:
			self.clip_grads(runner.model.parameters())
		runner.optimizer.step()
	
		
			
	