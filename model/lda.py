import numpy as np
from scipy.special import digamma, gamma
import math
from utils.dirichlet import pdf as dir_pdf, sample as dir_sample
from scipy.sparse import coo_matrix
from utils.log import logsum

EM_CONVERGED = 1e-6
EM_MAX_ITER = 10
VAR_MAX_ITER = 20
VAR_CONVERGED = 1e-4

"""
	LDA_VB: {
		alpha, beta, K, V
	}
"""
class LDA_VB:
	def __init__(self, num_topics, num_terms):
		self.num_topics = num_topics # Number of topics: K
		self.num_terms = num_terms # Number of terms: V
		self.alpha = 1.0 # Alpha: [1,1,..,1]: K-array
		self.log_beta = np.zeros((num_topics, num_terms)) # Logarit of beta: KxV
		self.EM_CONVERGED = EM_CONVERGED
		self.EM_MAX_ITER = EM_MAX_ITER
		self.VAR_MAX_ITER = VAR_MAX_ITER
		self.VAR_CONVERGED = VAR_CONVERGED
		
	# Estimate model parameters with the EM algorithm.	
	def fit(self, corpus):
		# Sufficient stats
		self.ss = SufficientStatistic(self, corpus)
		# EM
		self._em(corpus)

	# Initialize variational parameters
	def _init_params(self, corpus):
		# Variational parameter gamma: DxK
		var_gamma = np.zeros((corpus.num_docs, self.num_topics))

		# Variational parameter phi: NMAX x K
		phi = np.zeros((corpus.max_length, self.num_topics))

		return var_gamma, phi

	# EM algorithm, with paramaters initialized	
	def _em(self, corpus):
		# Init variational parameters
		var_gamma, phi = self._init_params(corpus)

		i = 0
		old_likelihood = 0
		converged = 1
		while (converged < 0 or converged > self.EM_CONVERGED or \
				i <= 2) and i < self.EM_MAX_ITER:
			i += 1
			print '*** EM iteration %d ****\n' % i
			likelihood = 0
			self.ss.zero_init()

			# E step
			print 'E'
			for d in range(corpus.num_docs):
				print 'Document %d\n' % d
				likelihood += self._doc_e_step(corpus.docs[d],\
						var_gamma[d], phi)
			print 'M'
			self._maximization()
			# M step
			# Check for convergence
			print "EM Likelihood: %f" % likelihood
			if old_likelihood == 0:
				converged = 1
			else: 
				converged = (old_likelihood - likelihood) / old_likelihood
 			if converged < 0:
				self.VAR_MAX_ITER *= 2
			old_likelihood = likelihood

	# Document E step		
	def _doc_e_step(self, doc, gamma, phi):
		likelihood = self._inference(doc, gamma, phi)
		gamma_sum = 0
		for k in range(self.num_topics):
			gamma_sum += gamma[k]
			self.ss.alpha_suffstats += digamma(gamma[k])
		self.ss.alpha_suffstats += self.num_topics \
				* digamma(gamma_sum)	
		for n in range(doc['length']):
			for k in range(self.num_topics):
				t = doc['counts'][n] * phi[n][k]
				self.ss.topic_word[k][doc['words'][n]] += t 
				self.ss.topic_total[k] += t
		self.ss.num_docs += 1
		
		return likelihood	

	# Variational inference
	def _inference(self, doc, var_gamma, phi):
		converged = 1
		phi_sum = 0
		likelihood = 0
		old_likelihood = 0
		old_phi = np.zeros(self.num_topics)
		digamma_gam = np.zeros(self.num_topics)
		# compute posterior dirichlet
		for k in range(self.num_topics):
			var_gamma[k] = self.alpha + 1. * \
					doc['total'] / self.num_topics
			digamma_gam[k] = digamma(var_gamma[k])
			for n in range(doc['length']):
				phi[n][k] = 1. / self.num_topics

		var_i = 0
		while converged > self.VAR_CONVERGED and (var_i < self.VAR_MAX_ITER\
				or self.VAR_MAX_ITER == -1):
			var_i += 1
			for n in range(doc['length']):
				phi_sum = 0
				for k in range(self.num_topics):
					old_phi[k] = phi[n][k]
					phi[n][k] = digamma_gam[k] +\
							self.log_beta[k][doc['words'][n]]
					phi_sum = logsum(phi_sum, phi[n][k])
				for k in range(self.num_topics):
					phi[n][k] = np.exp(phi[n][k] - phi_sum)
					var_gamma[k] += doc['counts'][n] *\
							(phi[n][k] - old_phi[k])
					digamma_gam[k] = digamma(var_gamma[k])		
			likelihood = self._compute_likelihood(doc, phi, var_gamma)	
			# print 'Likelihood: %f' %likelihood	
			if old_likelihood == 0:
				converged = 1
			else:
				converged = (old_likelihood - likelihood) / old_likelihood
			old_likelihood = likelihood

		return likelihood	
	
	# Compute likelihood bound	
	def _compute_likelihood(self, doc, phi, var_gamma):
		likelihood = 0
		digsum = 0
		var_gamma_sum = 0
		dig = np.zeros(self.num_topics)

		for k in range(self.num_topics):
			dig[k] = digamma(var_gamma[k])
			var_gamma_sum += var_gamma[k]
		digsum = digamma(var_gamma_sum)
		likelihood = math.lgamma(self.alpha * self.num_topics) \
				- self.num_topics * math.lgamma(self.alpha) \
				- math.lgamma(var_gamma_sum)
		for k in range(self.num_topics):
			likelihood += (self.alpha - 1) * (dig[k] - digsum) \
					+ math.lgamma(var_gamma[k]) \
					- (var_gamma[k] - 1) * (dig[k] - digsum)
			for n in range(doc['length']):
				if phi[n][k] > 0:
					likelihood += doc['counts'][n] \
							* (phi[n][k] * ((dig[k] - digsum) - np.log(phi[n][k]) \
									+ self.log_beta[k][doc['words'][n]]))	
		return likelihood

	# Maximization 
	def _maximization(self):
		for k in range(self.num_topics):
			for w in range(self.num_terms):
				if self.ss.topic_word[k][w] > 0:
					self.log_beta[k][w] = np.log(self.ss.topic_word[k][w]) \
							- np.log(self.ss.topic_total[k])
				else:
					self.log_beta[k][w] = -100

	# Get parameters for this estimator.
	def get_params(self):
		return self.alpha, self.log_beta
		# return self._alpha, self._beta, self._phi, self._gamma

	# Predict the labels for the data samples in X using trained model.
	def predict(self):
		pass

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self.log_beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)

"""
	Sufficient statistic: {
		topic_total,
		topic_word,
		alpha_suffstats,
		num_docs
	}
"""

class SufficientStatistic:
	def __init__(self, lda_model, corpus):
		self.lda_model = lda_model
		self.num_docs = corpus.num_docs
		self.topic_total = np.zeros((lda_model.num_topics))
		self.topic_word = np.zeros((lda_model.num_topics, \
				lda_model.num_terms))

	def zero_init(self):
		self.alpha_suffstats = 0.
		self.topic_total = np.zeros((self.lda_model.num_topics))
		self.topic_word = np.zeros((self.lda_model.num_topics, \
				self.lda_model.num_terms))