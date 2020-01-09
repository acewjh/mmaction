import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchRankingLoss(nn.Module):
	def __init__(self, margin=2.0):
		super(BatchRankingLoss, self).__init__()
		self.margin = margin
	
	def forward(self, preds, labels):
		n = preds.size(0)
		loss = 0.0
		for i in range(n):
			for j in range(i + 1, n):
				loss += F.relu(-(preds[j] - preds[i]) * torch.sign(labels[j] - labels[i]) + self.margin)
		
		return loss


# class BatchRankingMSE_Loss(nn.Module):
# 	def __init__(self, alpha=None, beta=None):
# 		super(BatchRankingMSE_Loss, self).__init__()
# 		self.alpha = alpha
# 		self.beta = beta
#
# 	def forward(self, preds, labels):
# 		mse = nn.MSELoss()(preds, labels)
# 		ranking = BatchRankingLoss()(preds, labels)
# 		alpha = self.alpha if self.alpha is not None else 1.0
# 		if not self.beta is None:
# 			beta = self.beta
# 		else:
# 			if preds.size(0) == 1:
# 				beta = 1.0
# 			else:
# 				beta = 2.0 / (preds.size(0) * (preds.size(0) - 1))
# 		loss = alpha * mse + beta * ranking
#
# 		return loss


class BatchRankingMSE_Loss(nn.Module):
	def __init__(self, alpha=None, beta=None):
		super(BatchRankingMSE_Loss, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.cache_alpha = 1.0
		self.cache_beta = 1.0

	def forward(self, preds, labels, is_val=False):
		mse = nn.MSELoss()(preds, labels)
		ranking = BatchRankingLoss()(preds, labels)
		if not is_val:
			alpha = self.alpha if self.alpha is not None else 1.0
			if not self.beta is None:
				beta = self.beta
			else:
				with torch.no_grad():
					g1 = torch.norm(torch.autograd.grad(mse, preds, retain_graph=True)[0], 2)
					g2 = torch.norm(torch.autograd.grad(ranking, preds, retain_graph=True)[0], 2)
					beta = g1 / (g2 + 0.0001)
			if not (self.alpha is None or self.beta is None):
				s = alpha + beta
				alpha = alpha / s * 2.0
				beta = beta / s * 2.0
			self.cache_alpha = alpha
			self.cache_beta = beta
		else:
			alpha = self.cache_alpha
			beta = self.cache_beta

		loss = alpha * mse + beta * ranking

		return loss

class MultiTaskGradNormLoss(nn.Module):
	def __init__(self, loss_funcs, alpha=0.12):
		assert len(loss_funcs) > 1
		super(MultiTaskGradNormLoss, self).__init__()
		self.loss_funcs = nn.ModuleList(loss_funcs)
		self.alpha = alpha
		self.task_num = len(loss_funcs)
		self.loss_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(self.task_num)])
		self.loss_iter_0 = nn.ParameterList()
		self.record_first = False

	def forward(self, preds, labels):
		loss = 0.0
		task_loss = []
		grad_norms = []
		inv_training_rates = []
		for t in range(self.task_num):
			loss_t = self.loss_weights[t]*self.loss_funcs[t](preds, labels)
			loss += loss_t
			if not self.record_first:
				self.loss_iter_0.append(nn.Parameter(loss_t.data.cuda(loss_t.device), requires_grad=False))
			task_loss.append(loss_t)
			G_t = torch.autograd.grad(task_loss[t], preds, retain_graph=True, create_graph=True)
			G_t = torch.norm(G_t[0], 2)
			grad_norms.append(G_t)
			L_t = loss_t/self.loss_iter_0[t]
			inv_training_rates.append(L_t)
		self.record_first = True
		loss = loss/self.task_num
		G_avg = sum(grad_norms)/self.task_num
		L_avg = sum(inv_training_rates)/self.task_num
		L_grad = 0.0
		for t in range(self.task_num):
			r_t = inv_training_rates[t]/L_avg
			C_t = G_avg*r_t**self.alpha
			C_t = C_t.detach()
			L_grad += F.l1_loss(grad_norms[t], C_t)

		return loss, L_grad

	def renormalize(self):
		with torch.no_grad():
			coef = self.task_num / sum(self.loss_weights)
		for t in range(self.task_num):
			self.loss_weights[t] = nn.Parameter(coef*self.loss_weights[t])

class RankingMSE_GradNorm(nn.Module):
	def __init__(self, **kwargs):
		super(RankingMSE_GradNorm, self).__init__()
		self.loss_func = MultiTaskGradNormLoss((BatchRankingLoss(), nn.MSELoss()), **kwargs)

	def forward(self, preds, labels):
		return self.loss_func(preds, labels)

	def renormalize(self):
		self.loss_func.renormalize()

class BatchRankingPairLoss(nn.Module):
	def __init__(self, margin=2.0):
		super(BatchRankingPairLoss, self).__init__()
		self.margin = margin
	
	def forward(self, preds, labels):
		preds = preds.view(-1, 2)
		loss = torch.mean(F.relu((preds[:, 1] - preds[:, 0]) * torch.sign(labels[:, 0] - labels[:, 1]) + self.margin))
		
		return loss

class CCCLoss(nn.Module):
	def __init__(self):
		super(CCCLoss, self).__init__()
	
	def forward(self, preds, labels):
		m1 = torch.mean(preds)
		s1 = torch.var(preds)
		m2 = torch.mean(labels)
		s2 = torch.var(labels)
		convar = torch.mean((preds - m1) * (labels - m2))
		ccc = 2 * convar / (s1 ** 2 + s2 ** 2 + (m1 - m2) ** 2)
		
		return ccc


class MSE_CCCLoss(nn.Module):
	def __init__(self, alpha=1.0, beta=1.0):
		super(MSE_CCCLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.mse = nn.MSELoss()
		self.ccc = CCCLoss()
	
	def forward(self, preds, labels):
		loss = self.alpha * self.mse(preds, labels) + self.beta * self.ccc(preds, labels)
		
		return loss


class MultistreamLoss(nn.Module):
	def __init__(self, n_stream, branch_weight=None, alpha=None, beta=None):
		super(MultistreamLoss, self).__init__()
		if branch_weight is None:
			self.branch_weight = [1.0 for x in range(n_stream)]
		else:
			self.branch_weight = branch_weight
		branch_loss = []
		for i in range(n_stream):
			branch_loss.append(BatchRankingLoss())
		self.branch_loss_funcs = nn.ModuleList(branch_loss)
		self.pred_loss_func = BatchRankingMSE_Loss()
		self.alpha = alpha if not alpha is None else 1 / n_stream
		self.beta = beta if not beta is None else 1.0
	
	def forward(self, *input):
		scores, pred, label = input
		branch_loss = 0.0
		for i, s in enumerate(scores):
			branch_loss = branch_loss + self.branch_weight[i] * self.branch_loss_funcs[i](s, label)
		pred_loss = self.pred_loss_func(pred, label)
		loss = self.alpha * branch_loss + self.beta * pred_loss
		
		return loss