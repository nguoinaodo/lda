import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from utils import normalize
from document import Document
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


class OnlineLDAVB:
	def __init__(self):
		self.var_converged = 1e-6
		self.predictive_ratio = .8
		self.var_max_iter = 50
		self.em_max_iter = 2
		self.em_converged = 1e-4
		self.batch_size = 100

	# Set parameters	
	def set_params(self, alpha=False, beta=False, tau0=False, kappa=False, eta=False, \
				K=False, V=False, log=None, plotfile=None, predictive_ratio=None,
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
		# Plot
		if plotfile:
			self.plotfile = plotfile
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
	def _init_beta(self, W, D):
		# Multinomial parameter beta: KxV
		self.beta = normalize(np.random.gamma(100, 1./100, (self.K, self.V)), axis=1)

	def _init_beta_corpus(self, W, D):
		self.beta = np.zeros((self.K, self.V))
		num_doc_per_topic = 5

		for i in range(num_doc_per_topic):
		    rand_index = np.random.permutation(D)
		    for k in range(self.K):
		        d = rand_index[k]
		        doc = W[d]
		        for n in range(doc.num_terms):
		            self.beta[k][doc.terms[n]] += doc.counts[n]
		    
		self.beta += 1
		self.beta = normalize(self.beta, axis=1)


	# Fit data	
	def fit(self, W, W_test=None):
		"""
			W: list of documents
			N_epoch: number of epoch
		"""	
		# Init beta
		self._init_beta(W, len(W))
		# Run EM
		self._em(W, W_test) 

	# EM with N epochs
	def _em(self, W, W_test=None):
		D = len(W)
		with open(self.log, 'a') as log:
			print '----------------------------------'
			print 'Number of documents: %d' % D
			print 'Number of topics: %d' % self.K
			print 'Number of terms: %d' % self.V
			log.write('---------------------------------\n')
			log.write('Online LDA:\n')
			log.write('Number of documents: %d\n' % len(W))
			log.write('Number of topics: %d\n' % self.K)
			log.write('Number of terms: %d\n' % self.V)
			log.write('Batch size: %d\n' % self.batch_size)
			log.write('alpha=%f\n' % self.alpha)
			log.write('tau0=%f\n' % self.tau0)
			log.write('kappa=%f\n' % self.kappa)
			log.write('eta=%f\n' % self.eta)
			log.write('var_converged=%f\n' % self.var_converged)
			log.write('var_max_iter=%d\n' % self.var_max_iter)
			log.write('----------------------------------\n')

			# Start time
			start = time.time()
			# Permutation
			random_ids = np.random.permutation(D)
			# For minibatch
			batchs = range(int(math.ceil(D/self.batch_size)))
			predictives = [] # predictive after each minibatch
			for t in batchs:
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
				# Predictive after each minibatch
				if W_test != None:
					preds = []
					for pair in W_test:
						pred = self.predictive(pair[0], pair[1])
						preds.append(pred)
					predictives.append(np.average(preds))

			# Time
			run_time = time.time() - start
			log.write('Run time: %f\n' % run_time)
			print 'Run time: %f' % run_time
			# Plot
			if len(predictives) > 0:
				self.plot_2d(batchs, predictives, 'Minibatch', 'Per-word log predictive')

	# Init variational parameters for each document
	def _doc_init_params(self, W_d):
		phi_d = np.ones((W_d.num_words, self.K)) / self.K
		gamma_d = (self.alpha + 1. * W_d.num_words / self.K) * np.ones(self.K)	
		return phi_d, gamma_d	

	# Estimate batch
	def _estimate(self, W, batch_ids):
		# Init sufficiency statistic for minibatch
		suff_stat = np.zeros(self.beta.shape)
		# For document in batch
		for d in batch_ids:
			# Estimate doc
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			# Update sufficiency statistic
			for j in range(W[d].num_words):
				for k in range(self.K):
					suff_stat[k][W_d[j]] += phi_d[j][k]
		return suff_stat

	def _estimate_doc(self, W, d):
		# Document flatten
		W_d = W[d].to_vector()	
		# Init variational parameters
		phi_d, gamma_d = self._doc_init_params(W[d])

		# Coordinate ascent
		old_gamma_d = gamma_d
		for i in range(self.var_max_iter):
			# Update phi
			phi_d = normalize(self.beta.T[W_d, :] * np.exp(digamma(gamma_d)), axis=1)
			# Update gamma
			gamma_d = self.alpha + np.sum(phi_d, axis=0)

			# Check convergence
			meanchange = np.mean(np.fabs(old_gamma_d - gamma_d))
			if meanchange < self.var_converged:
				break
			old_gamma_d = gamma_d
		return phi_d, gamma_d, W_d

	# Update global parameter
	def _maximize(self, suff_stat):
		return normalize(suff_stat, axis=1) + 1e-100

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
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			phi.append(phi_d)
			var_gamma.append(gamma_d)
		return phi, var_gamma	

	# Split observed - held-out
	def _split_observed_heldout(self, W):
		D = len(W)
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
		return W_obs, W_he

	# Predictive distribution
	def _predictive(self, W_obs, W_he):
		sum_log_prob = 0 # Sum of log of P(w_new|w_obs, W)
		num_new_words = 0 # Number of new words
		# Infer
		phi, var_gamma = self._infer(W_obs, len(W_obs))
		# Per-word log probability
		for d in range(len(W_he)):
			num_new_words += W_he[d].num_words
			for i in range(W_he[d].num_terms):
				sum_log_prob += W_he[d].counts[i] * np.log(1. * var_gamma[d].dot(self.beta[:, W_he[d].terms[i]]) /\
						np.sum(var_gamma[d]))
		result = 1. * sum_log_prob / num_new_words
		return result	

	def predictive(self, W):
		# Split 
		W_obs, W_he = self._split_observed_heldout(W)
		return self._predictive(W_obs, W_he)

	def predictive(self, W_obs, W_he):
		return self._predictive(W_obs, W_he)

	# Plot
	def plot_2d(self, x, y, xlabel=None, ylabel=None):
		plt.plot(x, y, 'b')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if self.plotfile:
			plt.savefig(self.plotfile)
		# plt.show()
		