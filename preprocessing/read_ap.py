import numpy as np

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
	for l in train:
		d = [] # document vector
		a = l.strip().split(' ')
		N = int(a[0]) # number of unique terms
		# Add word to doc
		for t in a[1:]:
			b = t.split(':')
			w = int(b[0]) # term
			n_w = int(b[1]) # number of occurrence
			for i in range(n_w):
				d.append(w)
		# Add doc to corpus
		documents.append(np.array(d))
	return documents

docs_train = to_documents(train)
docs_test = to_documents(test)
# docs = to_documents(lines)

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
