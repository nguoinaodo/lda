import numpy as np 
from model.lda_online import OnlineLDAVB
from utils.model import save_online_model, load_model

# Data
from preprocessing.read import read
from preprocessing.dictionary import read_vocab
W = read('dataset/dataset24k/gro/grolier-train.txt')
V, dic, inv_dic = read_vocab('dataset/dataset24k/gro/grolier-voca.txt')

# Init 
V = len(dic) # number of terms
count = 0
dirname = 'test_online_gro/'
for size in [1000]:
	for k in [100]:
		for alpha in [.1]:
			for kappa in [.5]:
				for tau0 in [64]:
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
