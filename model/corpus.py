class Corpus:
	def __init__(self, docs, num_terms):
		self.docs = docs
		self.num_docs = len(docs)
		self.num_terms = num_terms
		self.max_length = self._get_max_length()

	def _get_max_length(self):
		max_len = 0
		for d in self.docs: 
			if d['length'] > max_len:
				max_len = d['length']

		return max_len
