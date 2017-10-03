import numpy as np
from scipy.special import digamma, gamma
import math
from utils.dirichlet import pdf as dir_pdf, sample as dir_sample
from scipy.sparse import coo_matrix
from utils.log import logsum

EM_CONVERGED = 1e-6
EM_MAX_ITER = 100
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
		old_likelihood = 0,
		converged = 1
		while (converged < 0 or converged > EM_CONVERGED or \
				i <= 2) and i < EM_MAX_ITER:
			i += 1
			print '*** EM iteration %d ****\n' % i
			likelihood = 0
			self.ss.zero_init()

			# E step
			for d in range(corpus.num_docs):
				if d % 1000 == 0:
					print 'Document %d\n' % d
				likelihood += self._doc_e_step(corpus.docs[d],\
						var_gamma[d], phi)
			self._maximization()
			# M step
			# Check for convergence
			if old_likelihood == 0:
				converged = -1
			else: 
				converged = (old_likelihood - likelihood) / old_likelihood
 			if converged < 0:
				VAR_MAX_ITER *= 2
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
		while converged > VAR_CONVERGED and (var_i < VAR_MAX_ITER\
				or VAR_MAX_ITER == -1):
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
			print 'Likelihood: %f' %likelihood	
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
		likelihood = np.log(gamma(self.alpha * self.num_topics)) \
				- self.num_topics * np.log(gamma(self.alpha)) \
				- np.log(gamma(var_gamma_sum))
		print 'Likelihood1: %f' %likelihood
		for k in range(self.num_topics):
			likelihood += (self.alpha - 1) * (dig[k] - digsum) \
					+ np.log(gamma(var_gamma[k])) \
					- (var_gamma[k] - 1) * (dig[k] - digsum)
			print 'Likelihood2: %f' %likelihood
			for n in range(doc['length']):
				if phi[n][k] > 0:
					likelihood += doc['counts'][n] \
							* (phi[n][k] * ((dig[k] - digsum) - np.log(phi[n][k]) \
									+ self.log_beta[k][doc['words'][n]]))	
					print 'Likelihood3: %f' %likelihood		
		return likelihood


	# # Mean-fields algorithm
	# def _mean_fields(self, d):
	# 	while True:
	# 		N_d = self._W[d].shape[0]
	# 		# Update gamma
	# 		self._gamma[d] = self._alpha + np.sum(self._phi[d], axis = 0) # K
			
	# 		# Update phi
	# 		# print self._beta.T[self._W[d], :].shape
	# 		self._phi[d] = self._beta.T[self._W[d], :] * np.exp(digamma(self._gamma[d])) # NxK
	# 		# print self._phi[d].shape
	# 		self._phi[d] /= np.sum(self._phi[d], axis = 1).reshape(N_d, 1)

	# 		# Check convergence
	# 		# lower_bound = self._lower_bound()
	# 		# if math.fabs(lower_bound - self._old_lower_bound) < self._tol:
	# 		# 	break
	# 		# self._old_lower_bound  = lower_bound
	# 		# print "Lower bound: %f" % self._old_lower_bound

	# Maximization 
	def _maximization(self):
		for k in range(self.num_topics):
			for j in range(self.num_terms):
				if self.ss.topic_word[k][w] > 0:
					self.log_beta[k][w] = np.log(self.ss.topic_word[k][w]) \
							- np.log(self.ss.topic_total[k])
				else:
					self.log_beta[k][w] = -100

	# # Maximization
	# def _maximization(self):
	# 	self._beta = np.zeros((self._K, self._V))
	# 	for d in range(self._D):
	# 		N_d = self._W[d].shape[0]
	# 		# Sparse matrix
	# 		row = range(N_d)
	# 		col = self._W[d]
	# 		data = [1] * N_d
	# 		A = coo_matrix((data, (row, col)), shape=(N_d, self._V)) # NxV
	# 		B = self._phi[d].T * A
	# 		self._beta += B # KxN . NxV = KxV
	# 	# Normalize
	# 	self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)

	# # Calculate lower bound
	# def _lower_bound(self):
	# 	result = 0
	# 	for d in range(self._D):
	# 		# Eq log(P(theta|alpha))
	# 		A1 = (self._alpha - 1).dot(digamma(self._gamma[d]) - digamma(np.sum(self._gamma[d]))) # 1xK . Kx1 = 1
	# 		A2 = np.log(np.sum(self._alpha)) - np.sum(np.log(self._alpha))
	# 		A = A1 + A2
	# 		# SUMn Eq log(P(Zn|theta))
	# 		B = np.sum(self._phi[d].dot(digamma(self._gamma[d]) - digamma(np.sum(self._gamma[d]))))
	# 		# SUMn Eq log(P(Wn|Zn, beta))
	# 		C1 = (self._beta[:, self._W[d]]).T # NxK
	# 		C = np.sum(self._phi[d] * C1)
	# 		# Eq log(q(theta|gamma))
	# 		D1 = (self._gamma[d] - 1).dot(digamma(self._gamma[d]) - digamma(np.sum(self._gamma[d]))) # 1xK . Kx1 = 1
	# 		D2 = np.log(np.sum(self._gamma[d])) - np.sum(np.log(self._gamma[d]))
	# 		D = D1 + D2
	# 		# SUMn Eq log(q(Zn))
	# 		E = np.sum(self._phi[d] * np.log(self._phi[d]))
	# 		result += A + B + C - D - E
	# 	return result

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