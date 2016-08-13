# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:13:33 2014

@author: jon
"""
import scipy.io as sio
Erpgaussk01 = sio.loadmat('FinalData/E_rpgauss_k01_data.mat')
Erpgaussk02 = sio.loadmat('FinalData/E_rpgauss_k02_data.mat')
Erpk01 = sio.loadmat('FinalData/E_rp_k01_data.mat')
Erpk02 = sio.loadmat('FinalData/E_rp_k02_data.mat')

Espreadrpgaussk01 = sio.loadmat('FinalData/Espread_rpgauss_k01_data.mat')
Espreadrpgaussk02 = sio.loadmat('FinalData/Espread_rpgauss_k02_data.mat')
Espreadrpk01 = sio.loadmat('FinalData/Espread_rp_k01_data.mat')
Espreadrpk02 = sio.loadmat('FinalData/Espread_rp_k02_data.mat')


import matplotlib.pyplot as plt
alpha = 0.5
plt.hold
plt.figure(1)
plt.loglog(Espreadrpk01['data'][0,:],Espreadrpk01['data'][1,:],'o',
	markerfacecolor='r',markeredgecolor='k',alpha=alpha)
plt.loglog(Espreadrpk02['data'][0,:],Espreadrpk02['data'][1,:],'o',
	markerfacecolor='g',markeredgecolor='k',alpha=alpha)
plt.loglog(Espreadrpgaussk01['data'][0,:],Espreadrpgaussk01['data'][1,:],'o',
	markerfacecolor='k',markeredgecolor='b',alpha=alpha)
plt.loglog(Espreadrpgaussk02['data'][0,:],Espreadrpgaussk02['data'][1,:],'o',
	markerfacecolor='m',markeredgecolor='k',alpha=alpha)

plt.loglog(Erpk01['data'][0,:],Erpk01['data'][1,:],'s',
	markerfacecolor='r',markeredgecolor='k',alpha=alpha)
plt.loglog(Erpk02['data'][0,:],Erpk02['data'][1,:],'s',
	markerfacecolor='g',markeredgecolor='k',alpha=alpha)
plt.loglog(Erpgaussk01['data'][0,:],Erpgaussk01['data'][1,:],'s',
	markerfacecolor='k',markeredgecolor='k',alpha=alpha)
plt.loglog(Erpgaussk02['data'][0,:],Erpgaussk02['data'][1,:],'s',
	markerfacecolor='m',markeredgecolor='k',alpha=alpha)
plt.grid(True)
plt.legend([r'$\langle E \rangle$, $R_p$, $\mu = 0.1$',
			r'$\langle E \rangle$, $R_p$, $\mu = 0.2$',
			r'$\langle E \rangle$, $\langle R_p \rangle$, $\mu = 0.1$',
			r'$\langle E \rangle$, $\langle R_p \rangle$, $\mu = 0.2$',
			r'$E$, $R_p$, $\mu = 0.1$',
			r'$E$, $R_p$, $\mu = 0.2$',
			r'$E$, $\langle R_p \rangle$, $\mu = 0.1$',
			r'$E$, $\langle R_p \rangle$, $\mu = 0.2$'], loc=3)

plt.axvline(x=7.73, ymin=0., ymax = 1, linewidth=2, linestyle='--', color='k')
plt.annotate('F = 7.73 N\nEnsemble-average\ncrush limit', xy=(7.73, 0.15), xytext=(15, 0.25),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.xlabel('Force (N)')
plt.ylabel('Normalized probability')
#plt.legend(('With elasticity reduction factor',"Standard Young Modulus"))
plt.show()