f = open('dataset/ap/vocab.txt', 'r')

# Read lines
lines = f.readlines()
V = len(lines)

# Dictionary
dictionary = {}
inverse_dictionary = {}
for i in range(V):
	t = lines[i].strip()
	dictionary[t] = i
	inverse_dictionary[i] = t
