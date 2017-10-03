f = open('dataset/ap/vocab.txt', 'r')

# Read lines
lines = f.readlines()
V = len(lines)

# Dictionary
dictionary = {}
inverse_dictionary = {}
terms = []
for i in range(V):
	t = lines[i].strip()
	terms.append(t)
	dictionary[t] = i
	inverse_dictionary[i] = t
