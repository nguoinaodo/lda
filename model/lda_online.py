import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from utils import normalize
from document import Document

class OnlineLDAVB:
	def __init__(self):
		self.var_converged = 1e-6
		self.predictive_ratio = .8
		self.var_max_iter = 50
		self.em_max_iter = 5
		self.em_converged = 1e-4
		self.batch_size = 100

	# Set parameters	
	def set_params(self, alpha=False, beta=False, tau0=False, kappa=False, eta=False, \
				K=False, V=False, log=None, predictive_ratio=None,
				var_converged=None, var_max_iter=None, em_max_iter=None, em_converged=None,
				batch_size=None):
		# Dirichlet parameters of topics distribution 
		if alpha:
			self.alpha = alpha
		# Topic - term probability
		if beta:
			self.beta = beta
		# Slow down the early stop iterations of the algorithm
		if tau0:
			self.tau0 = tau0
		# Control the rate old values of beta are forgotten
		if kappa:
			self.kappa = kappa
		# Learning rate
		if eta: 
			self.eta = eta
		# Number of topics
		if K:
			self.K = K 
		# Dictionary size
		if V: 
			self.V = V
		# Log result
		if log:
			self.log = log
		# Predictive observed - held-out ratio
		if predictive_ratio:
			self.predictive_ratio = predictive_ratio
		# Convergence
		if var_converged: 
			self.var_converged = var_converged
		if var_max_iter:
			self.var_max_iter = var_max_iter
		if em_max_iter:
			self.em_max_iter = em_max_iter
		if em_converged:
			self.em_converged = em_converged
		if batch_size:
			self.batch_size = batch_size

	# Get parameters for this estimator.
	def get_params(self):
		return self.alpha, self.beta, self.tau0, self.kappa, self.eta

	# Init beta	
	def _init_beta(self):
		# Multinomial parameter beta: KxV
		return normalize(np.random.gamma(100, 1./100, (self.K, self.V)), axis=1)

	# Fit data	
	def fit(self, W):
		"""
			W: list of documents
			N_epoch: number of epoch
		"""	
		self.beta = self._init_beta()
		self._em(W) 

	# EM with N epochs
	def _em(self, W):
		D = len(W)
		with open(self.log, 'w') as log:
			print '----------------------------------'
			log.write('---------------------------------\n')
			log.write('Stochatic LDA:\n')
			log.write('Number of documents: %d\n' % D)
			print 'Number of documents: %d' % D
			print 'Number of topics: %d' % self.K
			print 'Number of terms: %d' % self.V
			log.write('Number of topics: %d\n' % self.K)
			log.write('Number of terms: %d\n' % self.V)
			log.write('EM max iter: %d\n' % self.em_max_iter)
			log.write('alpha=%f\n' % self.alpha)
			log.write('tau0=%f\n' % self.tau0)
			log.write('kappa=%f\n' % self.kappa)
			log.write('eta=%f\n' % self.eta)
			log.write('tolerance_var=%f\n\n' % self.var_converged)

			# Start time
			start = time.time()
			# Permutation
			random_ids = np.random.permutation(D)
			# For minibatch
			for t in range(int(math.ceil(D/self.batch_size))):
				print "Minibatch %d" % t
				log.write("Minibatch %d\n" % t)
				# Start minibatch time
				mb_start = time.time()
				# Batch documents id
				batch_ids = random_ids[t * self.batch_size: (t + 1) * self.batch_size]
				# Estimation for minibatch
				log.write('E\n')
				print 'E'
				suff_stat = self._estimate(W, batch_ids)
				# Update beta
				log.write('M\n') 
				print 'M'
				beta_star = self._maximize(suff_stat) # intermediate
				ro_t = (self.tau0 + t) ** (-self.kappa) # update weight
				self.beta = (1 - ro_t) * self.beta + ro_t * beta_star
				# Batch run time
				mb_run_time = time.time() - mb_start
				log.write('Minibatch run time: %f\n' % mb_run_time)
				print 'Minibatch run time: %f' % mb_run_time
			# Time
			run_time = time.time() - start
			log.write('Run time: %f\n' % run_time)
			print 'Run time: %f' % run_time
	
	# Estimate batch
	def _estimate(self, W, batch_ids):
		# Init sufficiency statistic for minibatch
		suff_stat = np.zeros(self.beta.shape)
		# Init beta
		beta = self._init_beta()
		batch_old_lower_bound = -1e12
		for it in range(self.em_max_iter):
			batch_lower_bound = 0
			# E step
			# For document in batch
			for d in batch_ids:
				# Document flatten
				W_d = W[d].to_vector()	
				# Init variational parameters
				phi_d = np.ones((W[d].num_words, self.K)) / self.K
				gamma_d = (self.alpha + 1. * W[d].num_words / self.K) * np.ones(self.K)
				# Init doc lower bound
				old_lower_bound = -1e12
				lower_bound = 0
				# Coordinate ascent
				for i in range(self.var_max_iter):
					# Update phi
					phi_d = normalize(beta.T[W_d, :] * np.exp(digamma(gamma_d)), axis=1)
					# Update gamma
					gamma_d = self.alpha + np.sum(phi_d, axis=0)
					# Document lower bound
					lower_bound = self._doc_lower_bound(W_d, phi_d, gamma_d, beta)
					# Check convergence
					converged = np.fabs((old_lower_bound - lower_bound) / old_lower_bound)
					if converged < self.var_converged:
						break
					old_lower_bound = lower_bound
				batch_lower_bound += lower_bound
				# Update sufficiency statistic
				for j in range(W[d].num_words):
					for k in range(self.K):
						suff_stat[k][j] += phi_d[j][k]
			# M step
			beta = self._maximize(suff_stat)
			# Check convergence
			converged = np.fabs((batch_old_lower_bound - batch_lower_bound) / batch_old_lower_bound)
			if converged < self.em_converged:
				break
			batch_old_lower_bound = batch_lower_bound
		return suff_stat

	# Update global parameter
	def _maximize(self, suff_stat):
		return normalize(suff_stat, axis=1) + 1e-100

	# Document lower bound
	def _doc_lower_bound(self, W_d, phi_d, gamma_d, beta):
		# Calculate
		sub_digamma = digamma(gamma_d) - digamma(np.sum(gamma_d))
		# Eq log(P(theta|alpha))
		A1 = gammaln(self.K * self.alpha) - self.K * gammaln(self.alpha)
		A = A1 + np.sum((self.alpha - 1) * sub_digamma) # A = 0
		# SUMn Eq log(P(Zn|theta))
		B = np.sum(phi_d.dot(sub_digamma))
		# SUMn Eq log(P(Wn|Zn, beta))
		C1 = np.nan_to_num(np.log((beta[:, W_d]).T)) # NxK
		C = np.sum(phi_d * C1)
		# Eq log(q(theta|gamma))
		D1 = (gamma_d - 1).dot(sub_digamma) # 1xK . Kx1 = 1
		D2 = gammaln(np.sum(gamma_d)) - np.sum(gammaln(gamma_d))
		D = D1 + D2
		# SUMn Eq log(q(Zn))
		E = np.sum(phi_d * np.nan_to_num(np.log(phi_d)))
		result = A + B + C - D - E
		# print 'Document lower bound time: %f' % (time.time() - start) 
		return result

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self.beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)	
















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
			while i < W[d].num_terms and 1. * count_obs / N_d < self.predictive_ratio:
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
				sum_log_prob += np.log(1. * var_gamma[d].dot(self.beta[:, W_he[d].terms[i]]) /\
						np.sum(var_gamma[d]))
				num_new_words += W_he[d].counts[i]
		result = 1. * sum_log_prob / num_new_words
		return result