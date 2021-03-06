from scipy.special import gamma
from scipy.stats import dirichlet
import numpy as np

def pdf(alpha, x):
	A = gamma(np.sum(alpha)) / (1. * np.prod(gamma(alpha)))
	B = np.prod(np.power(x, (np.subtract(alpha, 1))))
	return A * B 

def sample(alpha):
	d = dirichlet(alpha)
	return d.rvs()

# Test
a = dirichlet([2,2,2,2])
# print a.pdf([.1,.3,.5,.1])
# print a.rvs()