import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from utils import normalize
from document import Document

EM_MAX_ITER = 100
VAR_MAX_ITER = 20

class LDA_VB:
	def __init__(self, alpha):
		# self._K = K # Number of topics
		self._alpha = alpha # Dirichlet parameter: double
		# self._V = V # Vocab size
		self._tol_EM = 1e-5
		self._tol_var = 1e-6
		self._old_lower_bound = -999999999999
		self._predictive_ratio =  .75

	# Set parameters
	def set_params(self, alpha=None, beta=None, K=None, V=None, log=None,
			tol_EM=None, predictive_ratio=None):
		if alpha:
			self._alpha = alpha
		if beta:
			self._beta = beta
		if K:
			self._K = K
		if V:
			self._V = V	
		if log:
			self._log = log
		if tol_EM:
			self._tol_EM = tol_EM
		if predictive_ratio:
			self._predictive_ratio = predictive_ratio

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
		self._beta = np.random.gamma(100, 1./100, (self._K, self._V))
		self._beta = normalize(self._beta, axis=1)

	def _init_var_params(self, W, D):
		# Variational parameter phi: DxNdxK
		phi = []
		for d in range(D):
			N_d = W[d].num_words
			# p_d = np.random.gamma(100., 1./100, (N_d, self._K))
			# p_d = normalize(p_d, axis=1)
			p_d = 1.* np.ones((N_d, self._K)) / self._K
			phi.append(p_d)
		phi = np.array(phi) 
		
		# Variational parameter gamma: DxK
		var_gamma = np.ones((D, self._K))

		return phi, var_gamma

	# EM algorithm, with paramaters initialized	
	def _em(self, W, D, phi, var_gamma):
		with open(self._log, 'w') as log:
			log.write('LDA:\n')
			log.write('Number of document: %d\n' % D)
			log.write('Number of topic: %d\n' % self._K)
			log.write('Number of term: %d\n' % self._V)
			log.write('alpha=%f\n' % self._alpha)
			log.write('tolerance_EM=%f\n' % self._tol_EM)
			log.write('tolerance_var=%f\n\n' % self._tol_var)

			em_start = time.time()
			for i in range(EM_MAX_ITER):
				print "EM iterator number %d" % i
				log.write('\nEM iterator number %d\n' %i)
				# E step
				print "E%d" % i
				start = time.time()
				phi, var_gamma = self._estimation(W, D, phi, var_gamma)
				# M step
				print "M%d" %i
				self._maximization(W, D, phi, var_gamma)

				# Check convergence
				lower_bound = self._lower_bound(W, D, phi, var_gamma, log)
				converged = math.fabs((lower_bound - self._old_lower_bound) / self._old_lower_bound)
				log.write('EM converged: %f\n' % converged)
				if converged < self._tol_EM:
					break
				self._old_lower_bound = lower_bound
				print "EM time %f" % (time.time() - start)
				log.write("EM time %f\n" % (time.time() - start))
	 			print "EM Lower bound: %f" % self._old_lower_bound
	 			log.write("EM Lower bound: %f\n" % self._old_lower_bound)

	 		log.write('Runtime: %d\n' % (time.time() - em_start))
	 		log.write('Lower bound: %f\n' % self._old_lower_bound)	

 	# Estimation
 	def _estimation(self, W, D, phi, var_gamma):
 		for d in range(D):
 			phi, var_gamma = self._mean_fields(d, W, phi, var_gamma)
 		return phi, var_gamma
 			
	# Mean-fields algorithm
	def _mean_fields(self, d, W, phi, var_gamma):
		N_d = W[d].num_words
		W_d = W[d].to_vector()
		old_gamma_d = np.ones(self._K)
		old_phi_d = 1. * np.ones((N_d, self._K)) / self._K
		for i in range(VAR_MAX_ITER):
			# Update gamma
			var_gamma[d] = self._alpha + np.sum(phi[d], axis = 0) # K
			# Update phi
			a = self._beta.T[W_d, :]
			b = np.exp(digamma(var_gamma[d]))
			phi[d] = normalize(a * b, axis=1)
			# Check convergence
			# converged = np.average(np.fabs(old_gamma_d - var_gamma[d]))
			converged = np.average(np.fabs(old_phi_d - phi[d])) +\
					np.average(np.fabs(old_gamma_d - var_gamma[d]))
			if converged < 2 * self._tol_var: 
				break
			old_gamma_d = var_gamma[d]
			old_phi_d = phi[d]
		return phi, var_gamma

	# Maximization
	def _maximization(self, W, D, phi, var_gamma):
		self._beta = np.zeros((self._K, self._V))
		for d in range(D):
			N_d = W[d].num_words
			# Sparse matrix
			row = range(N_d)
			col = W[d].to_vector()
			data = [1] * N_d
			A = coo_matrix((data, (row, col)), shape=(N_d, self._V)) # NxV
			B = phi[d].T * A
			self._beta += B # KxN . NxV = KxV
		# Normalize
		self._beta = normalize(self._beta, axis=1)

	# Calculate lower bound
	def _lower_bound(self, W, D, phi, var_gamma, log=None):
		print 'Compute lower bound'
		result = 0
		t0 = time.time()
		for d in range(D):
			sub_digamma = digamma(var_gamma[d]) - digamma(np.sum(var_gamma[d]))
			# Eq log(P(theta|alpha))
			A1 = gammaln(self._K * self._alpha) - self._K * gammaln(self._alpha)
			A = A1 + np.sum((self._alpha - 1) * sub_digamma) # A = 0
			# SUMn Eq log(P(Zn|theta))
			B = np.sum(phi[d].dot(sub_digamma))
			# SUMn Eq log(P(Wn|Zn, beta))
			C1 = np.nan_to_num(np.log((self._beta[:, W[d].to_vector()]).T)) # NxK
			C = np.sum(phi[d] * C1)
			# Eq log(q(theta|gamma))
			D1 = (var_gamma[d] - 1).dot(sub_digamma) # 1xK . Kx1 = 1
			D2 = gammaln(np.sum(var_gamma[d])) - np.sum(gammaln(var_gamma[d]))
			D = D1 + D2
			# SUMn Eq log(q(Zn))
			E = np.sum(phi[d] * np.nan_to_num(np.log(phi[d])))
			result += A + B + C - D - E
		print "Lower bound time: %f" % (time.time() - t0)
		if log:
			log.write("Lower bound time: %f\n" % (time.time() - t0))	
		return result

	# Get parameters for this estimator.
	def get_params(self):
		return self._alpha, self._beta

	# Infer new corpus
	def _infer(self, W, D):
		phi, var_gamma = self._init_var_params(W, D)
		# Estimation
		phi, var_gamma = self._estimation(W, D, phi, var_gamma)
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
