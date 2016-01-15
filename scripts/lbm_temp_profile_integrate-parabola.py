# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from scipy import integrate
fig, ax = plt.subplots(1)

ystar = np.linspace(0,20, 201)
ystar2 = np.linspace(0,20,200)


# plot results from LBM
x = np.linspace(0,1,100)
df = pd.read_csv('lbm/file.csv')
temperature = df['temperature']
ax.plot(ystar,temperature,'k',linewidth=2,label="LBM")
ILBM = integrate.trapz(temperature,ystar)

Tmid = temperature[(len(temperature)-1)/2]

N = 10001
M = np.linspace(0,300,N)
erim1 = 1.e5
for i, m in enumerate(M):
	T1 = (1.-x**2)*(Tmid+m) + x**2*573
	T2 = np.zeros(len(T1))
	i=0
	for row in reversed(T1):
	    T2[i] = row
	    i+=1
	Tint = np.append(T2,T1)
	Iint = integrate.trapz(Tint,ystar2)
	er = np.abs((Iint - ILBM)/ILBM)
	if er > erim1:
		break
	else:
		erim1 = er
		Iintim1 = Iint
		Tintim1 = Tint
Iint = Iintim1
Tint = Tintim1
Tmidint = max(Tint)
TmaxDelta = Tmidint - Tmid
ax.plot(ystar2,Tint,'k--', linewidth=2, label='Integral fit')

plt.legend(loc='best')
# plt.title(r'$\int \,T_{lbm}$ dx = $\int\, T_{fit}$ dx')
plt.xlabel(r'x* = x/d$_p$')
plt.ylabel('Temperature (K)')
plt.savefig('figures/lbm-temp-profile_parabolic.png')


plt.show()
