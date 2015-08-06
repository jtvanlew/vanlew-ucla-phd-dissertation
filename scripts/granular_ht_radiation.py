
# coding: utf-8

# In[41]:

import numpy as np
import matplotlib.pyplot as plt


color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [207/255, 207/255, 207/255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255],
             [152/255, 223/255, 138/255],
             [95./255, 158./255, 209./255]
             ]

def k_graphite(T):
    T_fit = np.array([25,100,200,300,400,500,1000])
    k_s_fit = np.array([110,100,92,85,79,75,50])
    
    func = np.polyfit(T_fit, k_s_fit, 1)
    k_poly = np.poly1d(func)

    k_solid = k_poly(T)

    return k_solid

def zs(epsilon, kappa):
    B = 1.25 * ((1.-epsilon)/epsilon)**(1.055)

    ke_star = (1.-np.sqrt(1.-epsilon))+\
              (2.*np.sqrt(1.-epsilon))/(1.-B/kappa)*\
              ((((1.-1./kappa)*B)/(1.-B/kappa)**2)*np.log(kappa/B)-(B+1.)/2.-(B-1.)/(1.-B/kappa))
    return ke_star

def bb(epsilon, epsilon_r, T, d, k_s):
    T = T + 273
    sigma = 5.67E-8 # W m−2 K−4
    B = 1.25 * ((1.-epsilon)/epsilon)**(10./9)
    Lambda = k_s / (4* sigma * T**3 * d)
    lambda_e1 = (1-np.sqrt(1-epsilon))*epsilon
    lambda_e2 = (np.sqrt(1-epsilon))/(2./epsilon_r - 1.)
    lambda_e3 = (B+1.)/B
    lambda_e4 = (1/(1+1/((2./epsilon_r - 1.)*Lambda)))
    lambda_e = (lambda_e1 + lambda_e2*lambda_e3*lambda_e4)*4.*sigma*T**3*d
    return lambda_e

d = 0.001 #m
epsilon = 0.36
epsilon_r = 0.8

T = np.linspace(100,800,100)

k_g = 0.0025 * (T+273)**0.72 # W/mK
k_s = k_graphite(T)

kappa = k_s/k_g

k_zs = zs(epsilon, kappa)
k_bb = bb(epsilon, epsilon_r, T, d, k_s)/k_g

fig = plt.figure(1)
ax = fig.gca()
ax.grid(True)
ax.plot(T, k_zs, color=color_idx[0], label='Zehner-Schlünder', linewidth=2)
ax.plot(T, k_bb, color=color_idx[1], label='Breitbach-Bartels', linewidth=2)
# ax.plot(kappa, k_zs, label=, color=color_idx[3], linewidth=2)
# ax.plot(kappa, k_hsu_sc, label='Hsu et al, Sq. Cyl.', color=color_idx[4], linewidth=2)
# ax.plot(kappa, k_hsu_cc, label='Hsu et al, Circ. Cyl.', color=color_idx[5], linewidth=2)
# ax.plot(kappa, k_hsu_cubes, label='Hsu et al, Cubic', color=color_idx[6], linewidth=2)
# ax.plot(kappa, k_hsu_zs, label='Hsu et al, ZS corr.', color=color_idx[7], linewidth=2)
# ax.plot(kappa, k_zsb, label='Zehner-Bauer-Schlünder', color=color_idx[8], linewidth=2)
# ax.plot(kappa, k_bb, label='Breitbach-Bartels')


plt.xlabel(r'T ($^o$C)')
plt.ylabel(r"$k_{eff}/k_g$")
plt.xlim([min(T),max(T)])
plt.legend(loc='upper left')



plt.show()