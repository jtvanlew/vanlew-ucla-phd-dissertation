import numpy as np 
import matplotlib.pyplot as plt 

rstar_1 = np.linspace(0,1,100)

data = np.loadtxt('pebble_packing_radii.txt')
for row in data:
	n = data[:,0]
	rstar_2 = data[:,1]



plt.close('all')
plt.figure(1)
plt.xlabel(r'Dimensionless time ($t/\tau$)')
plt.ylabel(r'Dimensionless total heat removed, $Q^*$')
plt.title(r'Biot Number = %s'%(np.round(Bi,2)))
plt.plot(Fo, Q1, label = 'Lumped capacitance model (LC)')
plt.plot(Fo, Q2,  marker = 'o', label = 'LC with Jeffreson correction (JC)')
plt.plot(Fo, Q3, label = 'Exact solution with thermal gradient (TG)')
plt.ylim([0, 1])
plt.legend(loc='best')

plt.show()