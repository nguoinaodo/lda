import csv
import numpy as np

def save_model(model, filename):
	alpha, beta = model.get_params()
	with open(filename, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow([alpha])
		writer.writerows(beta)

def load_model(model, filename):
	with open(filename, 'r') as f:
		alpha = float(f.readline())
		beta = []
		lines = f.readlines()
		for l in lines:
			a = l.split(',')
			beta.append(a)
		beta = np.array(beta).astype(np.float)
	lda = model(alpha)
	lda.set_params(beta=beta)
	return lda

def save_online_model(model, filename):
	alpha, beta, tau0, kappa, eta = model.get_params()
	with open(filename, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow([alpha, tau0, kappa])
		writer.writerows(beta)
