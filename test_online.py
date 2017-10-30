import numpy as np 
from model.lda_online import OnlineLDAVB
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
alpha = [.01]#, 0.1, 1] # dirichlet parameter
batch_size = [50]
dirname = 'test_online/'
for size in batch_size:
	for k in K:
		for a in alpha:
			print 'Number of terms: %d' % V
			# print 'Number of documents: %d' % len(W_tr)
			print 'Number of documents: %d' % len(W)

			# Model
			lda = OnlineLDAVB()
			lda.set_params(alpha=a, K=k, V=V, kappa=.5, tau0=256, eta=.7,\
					log=dirname + 'lda_log' + str(count) + '.txt',\
					batch_size=size)
			# Fitting
			lda.fit(W)

			# Result
			top_idxs = lda.get_top_words_indexes()
			# predictive = lda.predictive(W)
			with open(dirname + 'lda_result' + str(count) + '.txt', 'w') as f:
				# s = 'Predictive: %f' % predictive
				# f.write(s)
				for i in range(len(top_idxs)):
					s = '\nTopic %d:' % i 
					for idx in top_idxs[i]:
						s += ' %s' % inv_dic[idx]
					f.write(s)

			# Save model
			save_stochatic_model(lda, dirname + 'model' + str(count) + '.csv')
			count += 1
