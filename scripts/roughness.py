import numpy as np
F_ns = np.linspace(0.1,50,100)
i=0
E_ij = 1./(2.*(1.-0.24**2.)/(60.e9)) #GPa
R_ij = 0.001/2. #m
k_ij = 2.4/2.

m = 0.2
H = 4.e9
sigma = np.linspace(0.1, 10, 100)/10**6

X, Y = np.meshgrid(F_ns, sigma)
a = ((3./4.) * R_ij*X/E_ij)**(1./3.)
delta_n = ((3./4)*(X)/(E_ij*np.sqrt(R_ij)))**(2./3)
H_j = np.zeros([len(F_ns),len(sigma)])
H_j = (1./((H/(E_ij*delta_n))**(0.96)*(Y/m)*(1./(1.72*k_ij*a**(1.04))) + 1./(2*k_ij*a)))/(2*k_ij*a)
# for F_n in F_ns:
# 	a = ((3./4.) * R_ij*F_n/E_ij)**(1./3.)
# 	delta_n = ((3./4)*(F_n)/(E_ij*np.sqrt(R_ij)))**(2./3)
# 	H_j[i,:] = (1./((H/(E_ij*delta_n))**(0.96)*(sigma/m)*(1./(1.72*k_ij*a**(1.04))) + 1./(2*k_ij*a)))/(2*k_ij*a)
# 	i+=1

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

plt.figure()
CS = plt.contourf(X, Y*10**6, H_j,20,cmap=plt.cm.rainbow)
cbar = plt.colorbar(CS)
#plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('Rms height (micron)')
plt.xlabel('Contact force (N)')
#plt.title(r"$H_j/H_H$")
plt.show()