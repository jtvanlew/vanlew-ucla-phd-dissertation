import numpy as np
import matplotlib.pyplot as plt
sigma = 5.67e-8

T1 = 750.+273
T2 = np.linspace(T1-300, T1, 100)

D     = 1./1000        # m
V     = 4./3.*np.pi*(D/2)**3 # m3
A     = 4*np.pi*(D/2)**2  # m2

Fij = 1./6

Q = A*sigma*(T1**4 - T2**4)

plt.plot((T2-T1),Q)
plt.xlabel('Temperature difference (K)')
plt.ylabel(r'$Q_{rad}$ (W)')
plt.show()