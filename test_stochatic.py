import numpy as np 
from model.lda_stochatic import StochaticLDA_VB
from utils.model import save_stochatic_model, load_model

# Data
from preprocessing.read_ap import docs_train as W_tr, docs_test as W_test
from preprocessing.dictionary import dictionary as dic, \
		inverse_dictionary as inv_dic

# Init 
K = 100 # number of topics
alpha = 1 # dirichlet parameter
V = len(dic) # number of terms
print 'Number of terms: %d' % V
print 'Number of documents: %d' % len(W_tr)

# Model
lda = StochaticLDA_VB()
lda.set_params(alpha=alpha, K=K, V=V, kappa=.5, tau0=256, eta=.5)

# Fitting
lda.fit(W_tr, N_epoch=5)

# Result
top_idxs = lda.get_top_words_indexes()
perplexity = lda.perplexity(W_test)
with open('lda_stochatic_result.txt', 'w') as f:
	s = 'Perplexity: %f' % perplexity
	f.write(s)
	for i in range(len(top_idxs)):
		s = '\nTopic %d:' % i 
		for idx in top_idxs[i]:
			s += ' %s' % inv_dic[idx]
		f.write(s)

# Save model
save_stochatic_model(lda, 'model_stochatic.csv')
