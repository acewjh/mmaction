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


class BatchRankingMSE_Loss(nn.Module):
	def __init__(self, alpha=None, beta=None):
		super(BatchRankingMSE_Loss, self).__init__()
		self.alpha = alpha
		self.beta = beta
	
	def forward(self, preds, labels):
		mse = nn.MSELoss()(preds, labels)
		ranking = BatchRankingLoss()(preds, labels)
		alpha = self.alpha if self.alpha is not None else 1.0
		if not self.beta is None:
			beta = self.beta
		else:
			if preds.size(0) == 1:
				beta = 1.0
			else:
				beta = 2.0 / (preds.size(0) * (preds.size(0) - 1))
		loss = alpha * mse + beta * ranking
		
		return loss


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