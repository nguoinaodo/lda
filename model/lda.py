import numpy as np
from scipy.special import digamma, gamma
import math
from utils.dirichlet import pdf as dir_pdf, sample as dir_sample
from scipy.sparse import coo_matrix
import time

EM_MAX_ITER = 3
VAR_MAX_ITER = 3

class LDA_VB:
	def __init__(self, K, alpha):
		self._K = K # Number of topics
		self._alpha = alpha # Dirichlet parameter: double
		self._tol = 10
		self._old_lower_bound = 0

	# Estimate model parameters with the EM algorithm.	
	def fit(self, W, V):
		self._W = W # Documents
		self._D = len(W) # Number of documents
		self._V = V # Dictionary length
		# Init parameters
		self._init_params()
		# EM
		self._em()

	# Initialize parameters
	def _init_params(self):
		# Multinomial parameter beta: KxV
		self._beta = np.random.rand(self._K, self._V)
		self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)
		# Variational parameter phi: DxNdxK
		self._phi = []
		for d in range(self._D):
			N_d = self._W[d].shape[0]
			p_d = np.random.rand(N_d, self._K)
			p_d = 1. * p_d / np.sum(p_d, axis = 1).reshape(N_d, 1)
			self._phi.append(p_d)
		self._phi = np.array(self._phi) 
		
		# Variational parameter gamma: DxK
		self._gamma = []
		for d in range(self._D):
			g_d = self._alpha + np.sum(self._phi[d], axis = 0)
			self._gamma.append(g_d)
		self._gamma = np.array(self._gamma)

		return self._beta, self._phi, self._gamma

	# EM algorithm, with paramaters initialized	
	def _em(self):
		for i in range(EM_MAX_ITER):
			# E step
			print "E%d" % i
			for d in range(self._D):
				# print "Ed%d" % d
				self._mean_fields(d)
			# M step
			print "M%d" %i
			self._maximization()

			# Check convergence
			lower_bound = self._lower_bound()
			if math.fabs(lower_bound - self._old_lower_bound) < self._tol:
				break
			self._old_lower_bound = lower_bound
			print "Lower bound: %f" % self._old_lower_bound

	# Mean-fields algorithm
	def _mean_fields(self, d):
		# print 'Mean field of document number %d' % d
		for i in range(VAR_MAX_ITER):
			N_d = self._W[d].shape[0]
			# Update gamma
			self._gamma[d] = self._alpha + np.sum(self._phi[d], axis = 0) # K
			
			# Update phi
			self._phi[d] = self._beta.T[self._W[d], :] * np.exp(digamma(self._gamma[d])) # NxK
			self._phi[d] /= np.sum(self._phi[d], axis = 1).reshape(N_d, 1)

			# Check convergence
			lower_bound = self._lower_bound()

			if math.fabs(lower_bound - self._old_lower_bound) < self._tol:
				break
			self._old_lower_bound  = lower_bound
			print "Lower bound var%d: %f" % (i, self._old_lower_bound)


	# Maximization
	def _maximization(self):
		self._beta = np.zeros((self._K, self._V))
		for d in range(self._D):
			N_d = self._W[d].shape[0]
			# Sparse matrix
			row = range(N_d)
			col = self._W[d]
			data = [1] * N_d
			A = coo_matrix((data, (row, col)), shape=(N_d, self._V)) # NxV
			B = self._phi[d].T * A
			self._beta += B # KxN . NxV = KxV
		# Normalize
		self._beta /= np.sum(self._beta, axis = 1).reshape(self._K, 1)

	# Calculate lower bound
	def _lower_bound(self):
		# print 'Compute lower bound'
		result = 0
		t0 = time.time()
		for d in range(self._D):
			# print "Document number %d" % d
			N_d = len(self._W[d])
			dig = digamma(self._gamma[d])
			digsum = digamma(np.sum(self._gamma[d]))
			# Eq log(P(theta|alpha))
			A = np.sum((self._alpha - 1) * (dig - digsum)) # A = 0
			# A=0
			# SUMn Eq log(P(Zn|theta))
			B = np.sum(self._phi[d].dot(dig - digsum))
			# SUMn Eq log(P(Wn|Zn, beta))
			C1 = np.log((self._beta[:, self._W[d]]).T) # NxK
			C = np.sum(self._phi[d] * C1)
			# Eq log(q(theta|gamma))
			D1 = (self._gamma[d] - 1).dot(dig - digsum) # 1xK . Kx1 = 1
			D2 = math.lgamma(np.sum(self._gamma[d]))
			for k in range(self._K):
				D2 -= math.lgamma(self._gamma[d][k])
			D = D1 + D2
			# SUMn Eq log(q(Zn))
			E = np.sum(self._phi[d] * np.log(self._phi[d]))
			result += A + B + C - D - E
		# print time.time() - t0
		
		return result

	# Get parameters for this estimator.
	def get_params(self):
		return self._alpha, self._beta, self._phi, self._gamma

	# Infer new corpus
	def infer(self):
		pass

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self._beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)

# Test
K = 10
alpha = np.array([1]*10)
lda = LDA_VB(10, alpha)
