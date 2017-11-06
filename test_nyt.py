import numpy as np 
from model.lda_online import OnlineLDAVB
from utils.model import save_online_model, load_model

# Data
from preprocessing.read import read
from preprocessing.dictionary import read_vocab
W = read('dataset/dataset24k/nyt/nytimes_row_200k.txt')
V, dic, inv_dic = read_vocab('dataset/dataset24k/nyt/vocab.nytimes.txt')
W_test = []
for i in range(1, 11):
	W_obs = read('dataset/dataset24k/nyt/data_test_' + str(i) + '_part_1.txt')
	W_he = read('dataset/dataset24k/nyt/data_test_' + str(i) + '_part_2.txt')
	W_test.append((W_obs, W_he))
# Init 
V = len(dic) # number of terms
count = 0
dirname = 'test_online_gro/'
for var_i in [50]:
	for size in [2000]:
		for k in [100]:
			for alpha in [.1]:
				for kappa in [.5]:
					for tau0 in [64]:
						# Model
						lda = OnlineLDAVB()
						lda.set_params(alpha=alpha, K=k, V=V, kappa=kappa, tau0=tau0, eta=.7,\
								log=dirname + 'lda_log' + str(count) + '.txt',\
								batch_size=size, plotfile=dirname + 'lda_plot' + str(count),\
								var_max_iter=var_i)
						# Fitting
						lda.fit(W, W_test)
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
						save_online_model(lda, dirname + 'model' + str(count) + '.csv')
						count += 1
