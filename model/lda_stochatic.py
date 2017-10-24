import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from utils import normalize
from document import Document

VAR_MAX_ITER = 20

class StochaticLDA_VB:
	def __init__(self):
		self._tol_var = 1e-6
		self._predictive_ratio = .8

	# Set parameters	
	def set_params(self, alpha=False, beta=False, tau0=False, kappa=False, eta=False, \
				K=False, V=False, log=None, predictive_ratio=None,
				tol_var=None):
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
		if log:
			self._log = log
		if predictive_ratio:
			self._predictive_ratio = predictive_ratio
		if tol_var: 
			self._tol_var = tol_var

	# Get parameters		
	def get_params(self):
		return self._alpha, self._beta, self._tau0, self._kappa, self._eta

	# Init beta	
	def _init_beta(self):
		# Multinomial parameter beta: KxV
		self._beta = np.random.gamma(100, 1./100, (self._K, self._V))
		self._beta = normalize(self._beta, axis=1)

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
		with open(self._log, 'w') as log:
			log.write('Stochatic LDA:\n')
			log.write('Number of documents: %d\n' % D)
			log.write('Number of topics: %d\n' % self._K)
			log.write('Number of terms: %d\n' % self._V)
			log.write('alpha=%f\n' % self._alpha)
			log.write('tau0=%f\n' % self._tau0)
			log.write('kappa=%f\n' % self._kappa)
			log.write('eta=%f\n' % self._eta)
			log.write('tolerance_var=%f\n\n' % self._tol_var)

			em_start = time.time()
			t = 0
			for e in range(N_epoch):
				print "Epoch number %d" % e
				log.write("Epoch number %d\n" % e)
				start = time.time()
				random_ids = np.random.permutation(D)
				for d in random_ids:
					phi_d, var_gamma_d = self._doc_init_params(W, d)
					phi_d, var_gamma_d = self._estimation_doc(W, D, d, phi_d, var_gamma_d)
					self._maximization_doc(W, D, d, phi_d, t)
					t += 1
				end = time.time() - start
				print "Epoch time %f" % end
				log.write("Epoch time %f" % end)

			log.write('Runtime: %d\n' % (time.time() - em_start))
			# Lower bound
			phi, var_gamma = self._infer(W, D)
			lower_bound = self._lower_bound(W, D, phi, var_gamma)
	 		log.write('Lower bound: %f\n' % lower_bound)	

	# Init params for each doc
	def _doc_init_params(self, W, d):
		phi_d = np.ones((W[d].num_words, self._K))
		var_gamma_d = (self._alpha + 1. * W[d].num_words / self._K) * np.ones(self._K)
		return phi_d, var_gamma_d

	# Estimation phi, gamma
	def _estimation_doc(self, W, D, d, phi_d, var_gamma_d):
		W_d = W[d].to_vector()
		old_lowerbound = self._doc_lower_bound(W, d, phi_d, var_gamma_d)
		for it in range(VAR_MAX_ITER):
			# Update phi
			phi_d = normalize(self._beta.T[W_d, :] * np.exp(digamma(var_gamma_d)), axis=1)
			# Update gamma
			var_gamma_d = self._alpha + np.sum(phi_d, axis = 0) # K
			# Check convergence
			doc_lower_bound = self._doc_lower_bound(W, d, phi_d, var_gamma_d)
			converged = np.fabs((old_lowerbound - doc_lower_bound) / old_lowerbound)
			if converged < self._tol_var:
				break
			old_lowerbound = doc_lower_bound
		return phi_d, var_gamma_d

	# Maximization: update beta
	def _maximization_doc(self, W, D, d, phi_d, t):
		N_d = W[d].num_words
		# Sparse matrix
		row = range(N_d)
		col = W[d].to_vector()
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
			phi_d, var_gamma_d = self._doc_init_params(W, d)
			self._estimation_doc(W, D, d, phi_d, var_gamma_d)
			phi.append(phi_d)
			var_gamma.append(var_gamma_d)
		return phi, var_gamma

	# Calculate lower bound
	def _lower_bound(self, W, D, phi, var_gamma):
		print 'Compute lower bound'
		result = 0
		t0 = time.time()
		for d in range(D):
			result += self._doc_lower_bound(W, d, phi[d], var_gamma[d])
		print "Time: %f" % (time.time() - t0)
		return result

	# Document lower bound
	def _doc_lower_bound(self, W, d, phi_d, var_gamma_d):
		start = time.time()
		# Calculate
		sub_digamma = digamma(var_gamma_d) - digamma(np.sum(var_gamma_d))
		# Eq log(P(theta|alpha))
		A1 = gammaln(self._K * self._alpha) - self._K * gammaln(self._alpha)
		A = A1 + np.sum((self._alpha - 1) * sub_digamma) # A = 0
		# SUMn Eq log(P(Zn|theta))
		B = np.sum(phi_d.dot(sub_digamma))
		# SUMn Eq log(P(Wn|Zn, beta))
		C1 = np.nan_to_num(np.log((self._beta[:, W[d].to_vector()]).T)) # NxK
		C = np.sum(phi_d * C1)
		# Eq log(q(theta|gamma))
		D1 = (var_gamma_d - 1).dot(sub_digamma) # 1xK . Kx1 = 1
		D2 = gammaln(np.sum(var_gamma_d)) - np.sum(gammaln(var_gamma_d))
		D = D1 + D2
		# SUMn Eq log(q(Zn))
		E = np.sum(phi_d * np.nan_to_num(np.log(phi_d)))
		result = A + B + C - D - E
		# print 'Document lower bound time: %f' % (time.time() - start) 
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
			c += d.num_words
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

	# Predictive distribution
	def predictive(self, W):
		D = len(W)
		phi, var_gamma = self._infer(W, D)
		sum_log_prob = 0
		num_new_words = 0
		W_obs = []
		W_he = []
		# Split document to observed and held-out
		for d in range(D):
			W_d = W[d].to_vector()
			N_d = W[d].num_words
			i = 0
			count_obs = 0
			while i < W[d].num_terms and 1. * count_obs / N_d < self._predictive_ratio:
				count_obs += W[d].counts[i]
				i += 1
			W_d_obs = Document(i, count_obs, W[d].terms[: i], W[d].counts[: i])
			W_d_he = Document(W[d].num_terms - i, N_d - count_obs, W[d].terms[i:], \
					W[d].counts[i:])
			W_obs.append(W_d_obs)
			W_he.append(W_d_he)
		# Infer
		phi, var_gamma = self._infer(W_obs, len(W_obs))
		# Per-word log probability
		for d in range(len(W_he)):
			for i in range(W_he[d].num_terms):
				sum_log_prob += np.log(1. * var_gamma[d].dot(self._beta[:, W_he[d].terms[i]]) /\
						np.sum(var_gamma[d]))
				num_new_words += W_he[d].counts[i]
		result = 1. * sum_log_prob / num_new_words
		return result