f = open('dataset/ap/vocab.txt', 'r')

# Read lines
lines = f.readlines()
num_terms = len(lines) # Number of terms

# Dictionary
dictionary = {}
inverse_dictionary = {}
for i in range(num_terms):
	t = lines[i].strip()
	dictionary[t] = i
	inverse_dictionary[i] = t
