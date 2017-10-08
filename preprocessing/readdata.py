import numpy as np

f = open('dataset/ap/ap.dat', 'r')

# Read lines
lines = f.readlines()

"""
	A document: {
		length: Number of unique terms
		words: Array of terms
		counts: Array of number of occurence of relative terms
		total: Number of words in document
	}
"""

# To documents
documents = []
for l in lines:
	d = {} # document vector
	a = l.strip().split(' ')
	d['length'] = int(a[0]) # number of unique terms
	d['words'] = []
	d['counts'] = []
	d['total'] = 0
	# Add word to doc
	for t in a[1:]:
		b = t.split(':')
		w = int(b[0]) # term
		n_w = int(b[1]) # number of occurrence
		
		d['words'].append(w)
		d['counts'].append(n_w)
		d['total'] += n_w
	# Add doc to corpus
	documents.append(d)


"""
	A corpus: {
		docs: [Document] (W),
		num_terms: Number of terms (V), 
		num_docs: Number of docs (D)
	}
"""

# To corpus
from dictionary import num_terms
from model.corpus import Corpus

corpus = Corpus(documents[:10], num_terms)
