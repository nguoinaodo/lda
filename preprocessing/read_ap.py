import numpy as np
from document import Document

f = open('dataset/ap/ap.dat', 'r')

# Read lines
lines = f.readlines()
D = len(lines)
D_train = 3 * D /4
D_test = D - D_train
train = lines[: D_train]
test = lines[D_train:]
# To documents
def to_documents(lines):
	documents = []
	for l in lines:
		a = l.strip().split(' ')
		num_terms = int(a[0]) # number of unique terms
		words = []
		counts = []
		num_words = 0
		# Add word to doc
		for t in a[1:]:
			b = t.split(':')
			w = int(b[0]) # term
			n_w = int(b[1]) # number of occurrence
			words.append(w)
			counts.append(n_w)
			num_words += n_w
		# Add doc to corpus
		doc = Document(num_terms, num_words, terms, counts)
		documents.append(doc)
	return documents

docs = to_documents(lines)
docs_train = to_documents(train)
docs_test = to_documents(test)

from dictionary import V
# To documents: sparse matrix
from scipy.sparse import coo_matrix

def to_sparse_matrix(lines):
	D = len(lines)
	row = []
	col = []
	data = []
	for d in range(D):
		a = lines[d].strip().split(' ')
		for t in a[1:]:
			b = t.split(':')
			term = int(b[0])
			count = int(b[1])
			row.append(d)
			col.append(term)
			data.append(count)

	return coo_matrix((data, (row, col)), shape=(D, V))

# sparse_docs_train = to_sparse_matrix(train)
# sparse_docs_test = to_sparse_matrix(test)
sparse_docs = to_sparse_matrix(lines)
