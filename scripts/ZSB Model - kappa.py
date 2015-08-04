
# coding: utf-8

# In[41]:

import numpy as np
import matplotlib.pyplot as plt


color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]
             ]
def lambda_fluid(epsilon, lambda_r):
    B = 1.25 * ((1.-epsilon)/epsilon)**(10./9)
    lambda_f1 = (1.-np.sqrt(1.-epsilon))
    lambda_f2 = (2.*np.sqrt(1.-epsilon))/(1.-lambda_r*B)
    lambda_f3 = (((1-lambda_r)*B)/(1-lambda_r*B)**2)*np.log(1./(lambda_r*B)) - (B+1.)/2. - (B-1.)/(1.-lambda_r*B)
    lambda_f = (lambda_f1 + lambda_f2*lambda_f3)
    return lambda_f

kappa = np.linspace(0.1,100000,100000)
lambda_r = 1./kappa

epsilon = 0.36

k_tot     = lambda_fluid(epsilon, lambda_r)


T = np.linspace(100,600,100)
k_gas = 0.0025 * (T+273)**0.72 # W/mK

T_fit = np.array([25,100,200,300,400,500,1000])
k_s_fit = np.array([110,100,92,85,79,75,50])
z = np.polyfit(T_fit, k_s_fit, 1)
k_s = np.poly1d(z)
k_solid = k_s(T)


k_exp_graphite = [1.251, 1.462, 1.559, 1.629, 1.686, 1.744]
T_exp_graphite = [135.6, 249, 339.1, 417, 484, 544]

k_exp_mod = [2.36, 2.21, 2.26, 2.35, 2.43, 2.52]

z2 = np.polyfit(T_exp_graphite, k_exp_graphite, 1)
k_exp_s = np.poly1d(z2)
k_exp_solid = k_exp_s(T)

k_gas_exp = 0.0025*(np.asarray(T_exp_graphite)+273)**0.72
k_solid_exp = k_s(T_exp_graphite)

k_star = k_exp_graphite/k_gas_exp
k_mod_star = k_exp_mod/k_gas_exp

kappa_exp = k_solid_exp/k_gas_exp



k_parallel = epsilon + (1-epsilon)*kappa
k_series = 1/ ( epsilon + (1-epsilon)/kappa )







# import csv
# kappa_dig = []
# keff_dig = []
# with open('digitized.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         kappa_dig.append(row[0])
#         keff_dig.append(row[1])

fig = plt.figure(1)
ax = fig.gca()
ax.grid(True)

#ax.plot(kappa, k_tot, label='ZS Model', color='k')

ax.plot(kappa, k_parallel, color=color_idx[1], label='Parallel', linewidth=2)
ax.plot(kappa, k_series, color=color_idx[0], label='Series', linewidth=2)
#ax.scatter(kappa_dig, keff_dig, label='lit. data', marker='o', edgecolors='k', facecolors='none', s=80)
#ax.scatter(kappa_exp, k_star, label='graphite exp.', marker='s', edgecolors='k', facecolors='none', s=80)
# the modified effective thermal conductivity given next is something done to appease Alice but doesn't have _much_
# significance to the real physics we're trying to study. It's a band-aid because the Koreans were surprised that
# we measured low conductivity values and rather than trying to understand why, she just simply wants our data to be
# giving larger values. I hate this.
# ax.scatter(kappa_exp, k_mod_star, label='graphite mod.', marker='x', edgecolors='k', facecolors='none', s=80)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$\kappa$')
plt.ylabel(r"$k_{eff}/k_g$")

plt.xlim([min(kappa),max(kappa)])
plt.legend(loc='upper left')



plt.show()