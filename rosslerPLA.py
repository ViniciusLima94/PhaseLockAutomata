'''
	Two coupled oscilator to show the Phase Lock Analyses.
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import hilbert
import seaborn as sns

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv

def simulate(eps = 0):

	# Parameters 
	D          = 1.0
	delta_t = 2*np.pi / 100.0
	T       = np.arange(0, 10000, delta_t)

	x1 = np.ones(len(T)) * np.random.rand()
	x2 = np.ones(len(T)) * np.random.rand()
	y1 = np.ones(len(T)) * np.random.rand()
	y2 = np.ones(len(T)) * np.random.rand()
	z1 = np.ones(len(T)) * np.random.rand()
	z2 = np.ones(len(T)) * np.random.rand()

	for i in range( len(T) - 1 ):
		x1[i+1] = x1[i] + delta_t*( -y1[i] - z1[i] + eps*(x2[i]-x1[i]) + np.random.normal(0,1)) 
		y1[i+1] = y1[i] + delta_t * (x1[i] + 0.15*y1[i])
		z1[i+1] = z1[i] + delta_t * (0.2 + z1[i]*(x1[i]-10.0)) 
		x2[i+1] = x2[i] + delta_t*( -y2[i] - z2[i] + eps*(x1[i]-x2[i]) + np.random.normal(0,1))
		y2[i+1] = y2[i] + delta_t * (x2[i] + 0.15*y2[i])
		z2[i+1] = z2[i] + delta_t * (0.2 + z2[i]*(x2[i]-10.0)) 


	h1 = hilbert(x1)
	h2 = hilbert(x2)
	phi1 = np.unwrap(np.angle(h1))
	phi2 = np.unwrap(np.angle(h2))
	psi = (phi1-phi2) / (2*np.pi)

	return psi

psi1 = simulate(eps=0)
psi2 = simulate(eps=0.04)
plt.subplot(1,2,1)
sns.distplot(psi1)
plt.title(r'uncoupled $\epsilon = 0$')
plt.ylabel(r'$\Psi$ count')
plt.subplot(1,2,2)
plt.title(r'coupled $\epsilon = 0.04$')
sns.distplot(psi2)