import numpy as np

f = open('dataset/ap/ap.dat', 'r')

# Read lines
lines = f.readlines()
D = len(lines)
# To documents
documents = []
for l in lines:
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

from dictionary import V
# To documents: sparse matrix
from scipy.sparse import coo_matrix

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

documents_1 = coo_matrix((data, (row, col)), shape=(D, V))