import numpy as np

f = open('dataset/ap/ap.dat', 'r')

# Read lines
lines = f.readlines()

# To documents
documents = []
for l in lines:
	d = [] # document vector
	a = l.strip().split(' ')
	N = int(a[0]) # number of words
	# Add word to doc
	for t in a[1:]:
		b = t.split(':')
		w = int(b[0]) # term
		n_w = int(b[1]) # number of occurrence
		for i in range(n_w):
			d.append(w)
	# Add doc to corpus
	documents.append(np.array(d))
