import numpy as np
from time import time

# Data
from preprocessing.read_ap import sparse_docs as W_tr
from preprocessing.dictionary import dictionary as dic, \
		inverse_dictionary as inv_dic, terms

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Model
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=100, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(W_tr)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
print_top_words(lda, terms, 20)
