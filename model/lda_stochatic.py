import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time

VAR_MAX_ITER = 20

class StochaticLDA_VB:
	def __init__(self):
		self._tol = 1e-5

	# Set parameters	
	def set_params(self, alpha=False, beta=False, tau0=False, kappa=False, eta=False, K=False, V=False):
		# Dirichlet parameters of topics distribution 
		if alpha:
			self._alpha = alpha
		# Topic - term probability
		if beta:
			self._beta = beta
		# Slow down the early stop iterations of the algorithm
		if tau0:
			self._tau0 = tau0
		# Control the rate old values of beta are forgotten
		if kappa:
			self._kappa = kappa
		# Learning rate
		if eta: 
			self._eta = eta
		# Number of topics
		if K:
			self._K = K 
		# Dictionary size
		if V: 
			self._V = V

	# Get parameters		
	def get_params(self):
		return self._alpha, self._beta, self._tau0, self._kappa, self._eta

	# Init beta	
	def _init_beta(self):
		# Multinomial parameter beta: KxV
		self._beta = np.random.rand(self._K, self._V)
		self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)

	# Fit data	
	def fit(self, W, N_epoch):
		"""
			W: list of documents
			N_epoch: number of epoch
		"""	
		D = len(W)
		self._init_beta()
		self._em(W, D, N_epoch)

	# EM with N epochs
	def _em(self, W, D, N_epoch):
		t = 0
		for e in range(N_epoch):
			print "Epoch number %d" % e
			start = time.time()
			random_ids = np.random.permutation(D)
			for d in random_ids:
				var_gamma_d, phi_d = self._init_var_params(W, D, d)
				var_gamma_d, phi_d = self._estimation_doc(W, D, d, var_gamma_d, phi_d)
				self._maximization_doc(W, D, d, phi_d, t)
				t += 1
			print "Epoch time %f" % (time.time() - start)

	# Init variationals params
	def _init_var_params(self, W, D, d):
		# gamma_d: K
		var_gamma_d = np.ones(self._K)
		# phi_d: NdxK
		N_d = len(W[d])
		phi_d = np.random.rand(N_d, self._K)
		phi_d = 1. * phi_d / np.sum(phi_d, axis = 1).reshape(N_d, 1)
		return var_gamma_d, phi_d

	# Estimation phi, gamma
	def _estimation_doc(self, W, D, d, var_gamma_d, phi_d):
		N_d = len(W[d])
		old_gamma_d = var_gamma_d
		for it in range(VAR_MAX_ITER):
			# Update gamma d
			var_gamma_d = self._alpha + np.sum(phi_d, axis=0) # K
			# Update phi d
			phi_d = self._beta.T[W[d], :] * np.exp(digamma(var_gamma_d)) # NxK
			phi_d /= np.sum(phi_d, axis=1).reshape(N_d, 1)
			# Check convergence
			converged = np.average(np.fabs(old_gamma_d - var_gamma_d))
			if converged < self._tol:
				break
			old_gamma_d = var_gamma_d
		return var_gamma_d, phi_d

	# Maximization: update beta
	def _maximization_doc(self, W, D, d, phi_d, t):
		N_d = len(W[d])
		# Sparse matrix
		row = range(N_d)
		col = W[d]
		data = [1] * N_d
		A = coo_matrix((data, (row, col)), shape=(N_d, self._V))
		beta_star = phi_d.T * A	
		ro_t = self._update_weight(t)
		self._beta = (1 - ro_t) * self._beta + ro_t * beta_star	

	# Update beta weight at time t
	def _update_weight(self, t):
		ro_t = (self._tau0 + t) ** (-self._kappa)
		return ro_t  	

	# Inference new docs
	def _infer(self, W, D):
		phi = []
		var_gamma = []
		for d in range(D):
			var_gamma_d, phi_d = self._init_var_params(W, D, d)
			self._estimation_doc(W, D, d, var_gamma_d, phi_d)
			phi.append(phi_d)
			var_gamma.append(var_gamma_d)
		return phi, var_gamma

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