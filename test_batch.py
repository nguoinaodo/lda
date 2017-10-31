import numpy as np 
from model.lda import LDA_VB
from utils.model import save_model, load_model

# Data
from preprocessing.read import read
from preprocessing.dictionary import read_vocab
W = read('dataset/ap/ap.dat')
V, dic, inv_dic = read_vocab('dataset/ap/vocab.txt')

# Init 
V = len(dic) # number of terms
count = 0
K = [100, 50] # number of topics
alpha = [1, 0.1, .01, 10] # dirichlet parameter
tol_EMs = [1e-4, 1e-5]
dirname = 'test_batch_result_new_ds/'
for k in K:
	for a in alpha:
		for t in tol_EMs:
			print 'Number of terms: %d' % V
			# print 'Number of documents: %d' % len(W_tr)
			print 'Number of documents: %d' % len(W)

			# Model
			lda = LDA_VB(a)
			lda.set_params(tol_EM=t, K=k, V=V, log=dirname + 'lda_log' + str(count) + '.txt')

			# Fitting
			# lda.fit(W_tr)
			lda.fit(W)

			# Result
			top_idxs = lda.get_top_words_indexes()
			# perplexity = lda.perplexity(W_test)
			with open(dirname + 'lda_result' + str(count) + '.txt', 'w') as f:
				# s = 'Perplexity: %f' % perplexity
				# f.write(s)
				for i in range(len(top_idxs)):
					s = '\nTopic %d:' % i 
					for idx in top_idxs[i]:
						s += ' %s' % inv_dic[idx]
					f.write(s)

			# Save model
			save_model(lda, dirname + 'model' + str(count) + '.csv')
			count += 1
