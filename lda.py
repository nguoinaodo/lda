import numpy as np
from scipy.special import digamma, gamma
import math

class LDA_VB:
	def __init__(self, K, alpha):
		self._K = K # Number of topics
		self._alpha = alpha # Dirichlet parameter: K
		self._tol = 1e-4

	# Estimate model parameters with the EM algorithm.	
	def fit(self, W, V):
		self._W = W # Documents
		self._D = W.shape[0] # Number of documents
		self._V = V # Dictionary length
		# Init parameters
		self._init_params()
		# EM
		self._em()

	# Initialize parameters
	def _init_params(self):
		self._beta = # Multinomial parameter: KxV
		self._phi = # Variational parameter: DxVxK
		self._gamma = # Variational parameter: DxK

	# EM algorithm, with paramaters initialized	
	def _em(self):
		while True:
			# E step
			for d in range(self._D):
				self._mean_fields(d)
			# M step
			self._maximization()
			# Check convergence
			

	# Mean-fields algorithm
	def _mean_fields(self, d):
		while True:
			# Update gamma
			self._gamma[d] = self._alpha + np.sum(self._phi, axis = 0) # K
			
			# Update phi
			self._phi = self._beta[:, self._W[d]] * np.exp(digamma(self._gamma)) # KxN
			self._phi /= 1. * np.sum(self._phi, axis = 0)

			# Check convergence
			lower_bound = self._lower_bound()
			if math.fabs(lower_bound - old_lower_bound) < tol:
				break
			old_lower_bound = lower_bound

	# Maximization
	def _maximization(self):
		for k in range(self._K):
			for j in range(V):
				self._beta[k, j] = 0
				for d in range(self._D):
					for n in range(len(self._W[d])):
						if self._W[d, n] == j:
							self._beta[k, j] += self._phi[d, n, k]

	# Calculate lower bound
	def _lower_bound(self):
		result = 0
		for d in range(self._D):
			# Eq log(P(theta|alpha))
			A1 = (self._alpha - 1).dot(digamma(self._gamma) - digamma(np.sum(self._gamma))) # 1xK . Kx1 = 1
			A2 = math.log(np.sum(self._alpha)) - np.sum(math.log(self._alpha))
			A = A1 + A2
			# SUMn Eq log(P(Zn|theta))
			B = np.sum(self._phi[d].dot(digamma(self._gamma) - digamma(np.sum(self._gamma))))
			# SUMn Eq log(P(Wn|Zn, beta))
			C1 = (self._beta[:, self._W[d]]).T # NxK
			C = np.sum(self._phi[d] * C1)
			# Eq log(q(theta|gamma))
			D1 = (self._gamma - 1).dot(digamma(self._gamma) - digamma(np.sum(self._gamma))) # 1xK . Kx1 = 1
			D2 = math.log(np.sum(self._gamma)) - np.sum(math.log(self._gamma))
			D = D1 + D2
			# SUMn Eq log(q(Zn))
			E = np.sum(self._phi * math.log(self._phi))
			result += A + B + C - D - E
		return result

	# Get parameters for this estimator.
	def get_params(self):
		pass 

	# Predict the labels for the data samples in X using trained model.
	def predict(self):
		pass 
