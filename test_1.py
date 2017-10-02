import numpy as np 
from model.lda import LDA_VB

# Data
from preprocessing.readdata import documents as W
from preprocessing.dictionary import dictionary as dic, \
		inverse_dictionary as inv_dic

# Init 
K = 100 # number of topics
alpha = np.array([1] * K) # dirichlet parameter
V = len(dic) # number of terms
print 'Number of terms: %d' % V
print 'Number of documents: %d' % len(W)

# Model
lda = LDA_VB(K, alpha)

# Fitting
lda.fit(W, V)

# Print top words
top_idxs = lda.get_top_words_indexes()
for i in range(len(top_idxs)):
	s = '\nTopic %d:' % i 
	for idx in top_idxs[i]:
		s += ' %s' % inv_dic[idx]
	print s	
