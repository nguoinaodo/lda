import numpy as np 
from model.lda_stochatic import StochaticLDA_VB
from utils.model import save_stochatic_model, load_model

# Data
from preprocessing.read_ap import docs_train as W_tr, docs_test as W_test,\
		docs as W
from preprocessing.dictionary import dictionary as dic, \
		inverse_dictionary as inv_dic

# Init 
V = len(dic) # number of terms
count = 0
K = [100] # number of topics
alpha = [.01, 0.1, 1] # dirichlet parameter
tol_var = [1e-6]
dirname = 'test_stochatic/'
for t in tol_var:
	for k in K:
		for a in alpha:
			print 'Number of terms: %d' % V
			# print 'Number of documents: %d' % len(W_tr)
			print 'Number of documents: %d' % len(W)

			# Model
			lda = StochaticLDA_VB()
			lda.set_params(tol_var=t, alpha=a, K=k, V=V, kappa=.5, tau0=256, eta=.5,\
					log=dirname + 'lda_log' + str(count) + '.txt')
			# Fitting
			lda.fit(W, N_epoch=5)

			# Result
			top_idxs = lda.get_top_words_indexes()
			predictive = lda.predictive(W)
			with open(dirname + 'lda_result' + str(count) + '.txt', 'w') as f:
				s = 'Predictive: %f' % predictive
				f.write(s)
				for i in range(len(top_idxs)):
					s = '\nTopic %d:' % i 
					for idx in top_idxs[i]:
						s += ' %s' % inv_dic[idx]
					f.write(s)

			# Save model
			save_stochatic_model(lda, dirname + 'model' + str(count) + '.csv')
			count += 1
