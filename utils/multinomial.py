import numpy as np 

# Sample multinomial distribution: p = [p1, p2, .., pK]
def sample(p):
	K = len(p)
	# Accumulate prob
	p0 = 0
	for i in range(K):
		p0 += p[i]
		p[i] = p0
	p = np.concatenate(([0], p))
	# Rand
	r = np.random.rand()
	# Search position
	i = 0
	j = K
	while i + 1 < j:
		if r >= p[i + 1]:
			i += 1
		if r < p[j - 1]:
			j -= 1		
	return j		

# Test
p = [.1, .1, .4, .4]                                  
print sample(p)



