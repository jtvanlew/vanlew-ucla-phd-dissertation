import numpy as np 
import matplotlib.pyplot as plt 

rstar_1 = np.linspace(0.2,1,100)

data = np.loadtxt('pebble_packing_radii.txt')
rstar_2 = {}
for row in data:
    n = data[:,0]
    rstar = data[:,1]
for i, n in enumerate(n):
    rstar_2[str(int(n))] = rstar[i]

N = np.round(1/rstar_1**3,0)
Vstar = []
Pf = []
for j, n in enumerate(N):
    n = str(int(n))
    if n in rstar_2:
        #print n + ' particle fragment(s)'
        #print 'r*1 = '+str(rstar_1[j])
        #print 'r*2 = '+str(rstar_2[n])
        Pf.append(n)
        Vstar.append((rstar_1[j]/rstar_2[n])**3)
    else:
        print 'radius ratio for '+ n + ' pebbles is not in the dictionary'

plt.plot(Pf, Vstar)
plt.figure(1)
plt.xlabel(r'Number of particles')
plt.ylabel(r'Dimensionless Volume')
plt.xlim([0, 72])
plt.legend(loc='best')

plt.show()