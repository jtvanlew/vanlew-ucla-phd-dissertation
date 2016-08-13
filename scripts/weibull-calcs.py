import numpy as np
import matplotlib.pyplot as plt

nu = 0.24
E =np.linspace(10e9, 100e9, 20)
Estar = 1/(2*((1-nu**2)/E))
Rstar = 0.00025


W = np.linspace(0.01, 1, 20) #mJ

X, Y = np.meshgrid(Estar, W)

Fcb = (4./3)*(15./8)**(3./5) * X**(2./5) * Rstar**(1./5)*(2*Y/1000)**(3./5)
plt.figure()
CS = plt.contour(X/10**9, Y, Fcb)
plt.clabel(CS, inline=1, fontsize=14)
plt.xlabel("E*, Pair Young Modulus (GPa)")
plt.ylabel("Micro-Strain Energy (mJ)")
plt.show()