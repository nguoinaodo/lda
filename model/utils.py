import numpy as np

def normalize(X, axis):
	if axis == 0:
		# Column nomalize
		X1 = 1. * X / np.sum(X, axis=0) 
	elif axis == 1:
		# Row normalize
		X1 = 1.* X / np.sum(X, axis=1).reshape(X.shape[0], 1)
	else:
		return X
	return X1
