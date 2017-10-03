import numpy as np

def logsum(loga, logb):
	if loga < logb:
		return logb + np.log(1 + np.exp(loga - logb))
	return loga + np.log(1 + np.exp(logb - loga))
