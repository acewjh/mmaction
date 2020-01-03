import numpy as np
from scipy.stats import spearmanr

def accuracy(pred, labels):
	return np.sum(pred == labels) / pred.shape[0]

def src(pred, labels):
	return spearmanr(pred, labels)[0]

def med(preds, labels):
	return np.mean(np.sqrt((preds - labels) ** 2))