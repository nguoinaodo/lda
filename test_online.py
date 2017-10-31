import numpy as np 
from model.lda_online import OnlineLDAVB
from utils.model import save_online_model, load_model

# Data
from preprocessing.read_ap import docs_train as W_tr, docs_test as W_test,\
		docs as W
from preprocessing.dictionary import dictionary as dic, \
		inverse_dictionary as inv_dic

# Init 
V = len(dic) # number of terms
count = 0
dirname = 'test_online/'
for size in [100]:
	for k in [100]:
		for alpha in [.01, .1]:
			for kappa in [.1, .5,.6,.7,.8,.9]:
				for tau0 in [0.9]
					# Model
					lda = OnlineLDAVB()
					lda.set_params(alpha=alpha, K=k, V=V, kappa=kappa, tau0=tau0, eta=.7,\
							log=dirname + 'lda_log' + str(count) + '.txt',\
							batch_size=size)
					# Fitting
					lda.fit(W)
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
					save_online_model(lda, dirname + 'model' + str(count) + '.csv')
					count += 1
