import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import hilbert
import seaborn as sns

def transition(p):
	if np.random.uniform(0, 1) < p:
		return 1
	else:
		return 0

def automata(couple = 1):

	if couple < 1:
		couple = 1
	elif couple > 100:
		couple = 100

	N = 10000000  # Number of steps
	p = 0.01   # Transition probability
	a1 = np.zeros(N)
	a2 = np.zeros(N)

	for i in range(N-1):
		if a1[i] == 0:
			if a2[i] == 1:
				a1[i+1] = transition(p * couple)
			elif a2[i] == 0:
				a1[i+1] = transition(p)
		if a2[i] == 0:
			if a1[i] == 1:
				a2[i+1] = transition(p * couple)
			elif a1[i] == 0:
				a2[i+1] = transition(p)
		if a1[i] == 1:
			a1[i+1] = 0
		if a2[i] == 1:
			a2[i+1] = 0
	return a1, a2

def sync(x1, x2):
	h1 = hilbert(x1)
	h2 = hilbert(x2)
	phi1 = np.unwrap(np.angle(h1))
	phi2 = np.unwrap(np.angle(h2))
	psi = (phi1-phi2) / (2*np.pi)
	n, x = np.histogram(psi, bins=40)
	return psi,x[:-1], n / np.sum(n).astype(float)

a1_async, a2_async = automata(couple = 1)
a1_sync, a2_sync = automata(couple = 50)

psi_async, x1, n1 = sync(a1_async, a2_async)
psi_sync, x2, n2 = sync(a1_sync, a2_sync)

plt.figure()
plt.subplot(1,2,1)
sns.distplot(psi_async, kde = False, norm_hist = True)
plt.title(r'uncoupled')
plt.ylabel(r'$\Psi$ count', fontsize = 20)
plt.subplot(1,2,2)
plt.title(r'coupled')
sns.distplot(psi_sync, kde = False, norm_hist=True)
plt.tight_layout()
plt.savefig('phase_distribution.pdf', dpi = 600)
