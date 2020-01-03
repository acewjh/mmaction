import numpy as np
from scipy.stats import spearmanr

def accuracy(pred, labels):
	return np.sum(pred == labels) / pred.shape[0]

def src(pred, labels):
	return spearmanr(pred, labels)[0]
