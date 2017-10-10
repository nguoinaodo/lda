import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
import sys
from scipy.sparse import coo_matrix
import time

EM_MAX_ITER = 100
VAR_MAX_ITER = 20

class LDA_VB:
	def __init__(self, alpha):
		# self._K = K # Number of topics
		self._alpha = alpha # Dirichlet parameter: double
		# self._V = V # Vocab size
		self._tol = 1e-4
		self._old_lower_bound = -999999999999

	# Set parameters
	def set_params(self, alpha=False, beta=False, K=False, V=False):
		if alpha:
			self._alpha = alpha
		if beta:
			self._beta = beta
		if K:
			self._K = K
		if V:
			self._V = V	

	# Estimate model parameters with the EM algorithm.	
	def fit(self, W):
		D = len(W) # Number of documents
		# Init parameters
		self._init_params()
		phi, var_gamma = self._init_var_params(W, D)
		# EM
		self._em(W, D, phi, var_gamma)

	# Initialize parameters
	def _init_params(self):
		# Multinomial parameter beta: KxV
		self._beta = np.random.rand(self._K, self._V)
		self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)

	def _init_var_params(self, W, D):
		# Variational parameter phi: DxNdxK
		phi = []
		for d in range(D):
			N_d = W[d].shape[0]
			p_d = np.random.rand(N_d, self._K)
			p_d = 1. * p_d / np.sum(p_d, axis = 1).reshape(N_d, 1)
			phi.append(p_d)
		phi = np.array(phi) 
		
		# Variational parameter gamma: DxK
		var_gamma = np.ones((D, self._K))

		return phi, var_gamma

	# EM algorithm, with paramaters initialized	
	def _em(self, W, D, phi, var_gamma):
		for i in range(EM_MAX_ITER):
			print "EM iterator number %d" % i
			# E step
			print "E%d" % i
			start = time.time()
			self._estimation(W, D, phi, var_gamma)
			# M step
			print "M%d" %i
			self._maximization(W, D, phi, var_gamma)

			# Check convergence
			lower_bound = self._lower_bound(W, D, phi, var_gamma)
			if math.fabs(lower_bound - self._old_lower_bound) < self._tol:
				break
			self._old_lower_bound = lower_bound
			print "EM time %f" % (time.time() - start)
 			print "EM Lower bound: %f" % self._old_lower_bound

 	# Estimation
 	def _estimation(self, W, D, phi, var_gamma):
 		for d in range(D):
 			self._mean_fields(d, W, phi, var_gamma)

	# Mean-fields algorithm
	def _mean_fields(self, d, W, phi, var_gamma):
		# print 'Mean field of document number %d' % d
		old_gamma_d = np.ones(self._K)
		for i in range(VAR_MAX_ITER):
			N_d = W[d].shape[0]
			# Update gamma
			var_gamma[d] = self._alpha + np.sum(phi[d], axis = 0) # K
			
			# Update phi
			phi[d] = self._beta.T[W[d], :] * np.exp(digamma(var_gamma[d])) # NxK
			phi[d] /= np.sum(phi[d], axis = 1).reshape(N_d, 1)

			# Check convergence
			converged = np.average(np.fabs(old_gamma_d - var_gamma[d]))
			if converged < self._tol: 
				break
			old_gamma_d = var_gamma[d]

	# Maximization
	def _maximization(self, W, D, phi, var_gamma):
		self._beta = np.zeros((self._K, self._V))
		for d in range(D):
			N_d = W[d].shape[0]
			# Sparse matrix
			row = range(N_d)
			col = W[d]
			data = [1] * N_d
			A = coo_matrix((data, (row, col)), shape=(N_d, self._V)) # NxV
			B = phi[d].T * A
			self._beta += B # KxN . NxV = KxV
		# Normalize
		self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)

	# Calculate lower bound
	def _lower_bound(self, W, D, phi, var_gamma):
		print 'Compute lower bound'
		result = 0
		t0 = time.time()
		for d in range(D):
			dig = digamma(var_gamma[d])
			digsum = digamma(np.sum(var_gamma[d]))
			# Eq log(P(theta|alpha))
			A = np.sum((self._alpha - 1) * (dig - digsum)) # A = 0
			# SUMn Eq log(P(Zn|theta))
			B = np.sum(phi[d].dot(dig - digsum))
			# SUMn Eq log(P(Wn|Zn, beta))
			C1 = np.log((self._beta[:, W[d]]).T) # NxK
			C = np.sum(phi[d] * C1)
			# Eq log(q(theta|gamma))
			D1 = (var_gamma[d] - 1).dot(dig - digsum) # 1xK . Kx1 = 1
			D2 = gammaln(np.sum(var_gamma[d])) - np.sum(gammaln(var_gamma[d]))
			D = D1 + D2
			# SUMn Eq log(q(Zn))
			E = np.sum(phi[d] * np.log(phi[d]))
			result += A + B + C - D - E
		print "Time: %f" % (time.time() - t0)
		
		return result

	# Get parameters for this estimator.
	def get_params(self):
		return self._alpha, self._beta

	# Infer new corpus
	def _infer(self, W, D):
		phi, var_gamma = self._init_var_params(W, D)
		# Estimation
		self._estimation(W, D, phi, var_gamma)
		return phi, var_gamma

	# Perplexity
	def perplexity(self, W):
		D = len(W) # number of documents
		phi, var_gamma = self._infer(W, D)
		# Lower bound likelihood
		lower_bound = self._lower_bound(W, D, phi, var_gamma)
		num_words = self._count_words(W)
		# Perplexity
		perplexity = np.exp(-lower_bound / num_words)
		return perplexity

	# Document topics
	def docs_topics(self, W):
		D = len(W)
		phi, var_gamma = self._infer(W, D)
		top_idxs = []
		for gamma_d in var_gamma:
			desc_idx = np.argsort(gamma_d)[::-1]
			top_idx = desc_idx[:5]
			top_idxs.append(top_idx)
		return np.array(top_idxs)

	# Count tokens in documents	
	def _count_words(self, W):
		c = 0
		for d in W:
			c += len(d)	
		return c

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self._beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)
