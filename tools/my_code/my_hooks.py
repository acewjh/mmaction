import os
import numpy as np
import os.path as osp
from collections import OrderedDict
from mmcv.runner.hooks import Hook
from .metrics import accuracy, src, med

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
		self.output_cache = OrderedDict()
	
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
		for mtr in metrics:
			if mtr == 'accuracy':
				self.metrics_funcs.append(accuracy)
			elif mtr == 'src':
				self.metrics_funcs.append(src)
			else:
				self.metrics_funcs.append(med)
			
	def after_train_epoch(self, runner):
		for i, mtr in enumerate(self.metrics):
			performance = self.metrics_funcs[i](runner.output_cache['output'], runner.output_cache['labels'])
			runner.log_buffer.update({self.mtr:performance})
			runner.log_buffer.average()
		
	def after_val_epoch(self, runner):
		self.after_train_epoch(runner)
		
class SaveOutputHook(Hook):
	def after_train_epoch(self, runner):
		self.save_output('train', runner, 'pred_epoch_{}.npy', runner.output_cache['output'])
		self.save_output('train', runner, 'labels_epoch_{}.npy', runner.output_cache['labels'])
		
	def after_val_epoch(self, runner):
		self.save_output('val', runner, 'pred_epoch_{}.npy', runner.output_cache['output'])
		if runner.epoch == 0:
			self.save_output('val', runner, 'labels_epoch_{}.npy', runner.output_cache['labels'])
		
	def save_output(self, mode, runner, tmpl, tensor):
		save_path = osp.join(runner.work_dir, 'output', mode, tmpl.format(runner.epoch + 1))
		if not osp.exists(osp.dirname(save_path)):
			os.makedirs(osp.dirname(save_path))
		np.save(save_path, tensor)
	
		
			
	