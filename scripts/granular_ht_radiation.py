
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

def k_N(T):
    k_N_fit = np.array([0.0098709,0.014550,0.018659,0.022399,0.025858,0.029106,0.032205,0.035205,0.038143,0.041042,0.043917,0.046771,0.049605,0.052414,0.055197,0.057949,0.060666])
    T_fit = np.linspace(100,900,len(k_N_fit))

    func = np.polyfit(T_fit, k_N_fit, 2)
    k_poly = np.poly1d(func)

    k_N = k_poly(T)
    return k_N

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

def load_digitized_csv(filename):
    import csv
    x = []
    y = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            x.append(row[0])
            y.append(row[1])
    return x, y

d = 0.06 #m
epsilon = 0.41
epsilon_r = 0.8

T = np.linspace(100,800,100)
k_s = k_graphite(T)


# Helium comparison to SANA data
k_g_He = 0.0025 * (T+273)**0.72 # W/mK
kappa_He = k_s/k_g_He
k_zs_He = zs(epsilon, kappa_He)*k_g_He/100
k_bb_He = bb(epsilon, epsilon_r, T, d, k_s)/100
k_tot_He = k_zs_He + k_bb_He
T_dig_He, k_dig_He = load_digitized_csv('SANA-digitized-He.csv')

fig = plt.figure(num=0, figsize=(12, 9), dpi=150, facecolor='w', edgecolor='k')
ax = fig.gca()
ax.grid(True)
#ax.plot(T, k_zs, color=color_idx[0], label='Zehner-Schlünder', linewidth=2)
ax.plot(T, k_zs_He, color=color_idx[0], label='Zehner-Schlunder', linewidth=2)
ax.plot(T, k_bb_He, color=color_idx[1], label='Breitbach-Bartels', linewidth=2)
ax.plot(T, k_tot_He, color=color_idx[2], label='Tot = ZS + BB', linewidth=2)
ax.scatter(T_dig_He, k_dig_He, color=color_idx[3], s=20, label='SANA Experimental Data (Helium)')

plt.xlabel(r'T ($^o$C)')
plt.ylabel(r"$k_{eff}/k_g$")
plt.xlim([min(T),max(T)])
plt.ylim(0, 0.3)
plt.legend(loc='upper left')
plt.savefig('../figures/keff-sana-he', bbox_inches='tight')





# Nitrogen comparison to SANA data
k_g_N = k_N(T) # W/mK
kappa_N = k_s/k_g_N
k_zs_N = zs(epsilon, kappa_N)*k_g_N/100
k_bb_N = bb(epsilon, epsilon_r, T, d, k_s)/100
k_tot_N = k_zs_N + k_bb_N
T_dig_N, k_dig_N = load_digitized_csv('SANA-digitized-N.csv')


fig = plt.figure(num=1, figsize=(12, 9), dpi=150, facecolor='w', edgecolor='k')
ax = fig.gca()
ax.grid(True)
#ax.plot(T, k_zs, color=color_idx[0], label='Zehner-Schlünder', linewidth=2)
ax.plot(T, k_zs_N, color=color_idx[0], label='Zehner-Schlunder', linewidth=2)
ax.plot(T, k_bb_N, color=color_idx[1], label='Breitbach-Bartels', linewidth=2)
ax.plot(T, k_tot_N, color=color_idx[2], label='Tot = ZS + BB', linewidth=2)
ax.scatter(T_dig_N, k_dig_N, color=color_idx[3], s=20, label='SANA Experimental Data (Nitrogen)')

plt.xlabel(r'T ($^o$C)')
plt.ylabel(r"$k_{eff}/k_g$")
plt.xlim([min(T),max(T)])
plt.ylim(0, 0.3)
plt.legend(loc='upper left')
plt.savefig('../figures/keff-sana-n', bbox_inches='tight')

plt.show()