
# coding: utf-8

# In[41]:

import numpy as np
import matplotlib.pyplot as plt


color_idx = [[0./255,   107./255, 164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255,  89./255,  89./255],
             [207./255, 207./255, 207./255],
             [200./255, 82./255,  0./255],
             [255./255, 152./255, 150./255],
             [152./255, 223./255, 138./255],
             [95./255,  158./255, 209./255]
             ]
def load_digitized_csv(filename):
    import csv
    x = []
    y = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            x.append(row[0])
            y.append(row[1])
    x = np.array([float(i) for i in x])
    y = np.array([float(i) for i in y])
    return x, y

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

def hsu_sc(epsilon, kappa):
    gamma_c = 0.01
    # some work to find gamma_a because it's given as a non-linear function
    # of gamma_c and epsilon.
    # First define the full function
    def F(gamma_a, gamma_c, epsilon):
        return (gamma_a**2 + 2*gamma_c*gamma_a*(1-gamma_a))-(1-epsilon)
    # now i can create function handles so myfunct only takes 1 argument
    myfunct = lambda x: F(x, gamma_c, epsilon)
    import scipy.optimize
    # find gamma_a with an initial guess
    gamma_a = scipy.optimize.broyden1(myfunct, 1., f_tol=1e-14)
    
    ke_star = gamma_a*gamma_c*kappa + \
              (gamma_a*(1-gamma_c))/(1+(1./kappa-1)*gamma_a) + \
              (1-gamma_a)/(1+(1./kappa-1)*gamma_a*gamma_c)
    return ke_star    

def hsu_cc(epsilon, kappa):
    gamma_c = 0.01
    theta_c = np.arcsin(gamma_c)
    # some work to find gamma_a because it's given as a non-linear function
    # of gamma_c and epsilon.
    # First define the full function
    def F(gamma_a, gamma_c, theta_c, epsilon):
        return 1-gamma_c*gamma_a - (gamma_a**2/2.)*(np.pi/2-2*theta_c)-epsilon
    # now i can create function handles so myfunct only takes 1 argument
    myfunct = lambda x: F(x, gamma_c, theta_c, epsilon)
    import scipy.optimize
    # find gamma_a with an initial guess
    gamma_a = scipy.optimize.broyden1(myfunct, 1., f_tol=1e-14)
    ke_star = np.zeros(len(kappa))
    for i, kappa in enumerate(kappa):
        if (1./kappa-1)*gamma_a < 1:
            ke_star[i] = gamma_c*gamma_a*kappa +\
                      (1-gamma_a*np.sqrt(1-gamma_c**2))/(gamma_a*gamma_c*(1./kappa-1)+1) + \
                      (kappa*(np.pi/2-2*theta_c))/(1-kappa) - \
                      (2*kappa)/((1.-kappa)*np.sqrt(1-(1/kappa-1)**2*gamma_a**2))*\
                      (np.arctan((np.tan(np.pi/4-theta_c/2)+(1/kappa-1)*gamma_a)/(np.sqrt(1-(1/kappa-1)**2*gamma_a**2)))-\
                        np.arctan((np.tan(theta_c/2)+(1/kappa-1)*gamma_a)/(np.sqrt(1-(1/kappa-1)**2*gamma_a**2))))
        elif (1./kappa-1)*gamma_a > 1:
            ke_star[i] = gamma_c*gamma_a*kappa + \
                         (1-gamma_a*np.sqrt(1-gamma_c**2))/(gamma_a*gamma_c*(1/kappa-1)+1)+\
                         (kappa*(np.pi/2-2*theta_c))/(1-kappa)-\
                         kappa/((1-kappa)*np.sqrt((1/kappa-1)**2*gamma_a**2-1))*\
                         (np.log((np.tan(np.pi/4-theta_c/2)+(1/kappa-1)*gamma_a-np.sqrt((1/kappa-1)**2*gamma_a**2-1))/(np.tan(np.pi/4-theta_c/2)+(1/kappa-1)*gamma_a+np.sqrt((1/kappa-1)**2*gamma_a**2-1)))-\
                            np.log((np.tan(theta_c/2)+(1/kappa-1)*gamma_a-np.sqrt((1/kappa-1)**2*gamma_a**2-1))/(np.tan(theta_c/2)+(1/kappa-1)*gamma_a+np.sqrt((1/kappa-1)**2*gamma_a**2-1))))
        else:
            ke_star[i] = (gamma_c*gamma_a**2)/(gamma_a+1)+(1-gamma_a*np.sqrt(1-gamma_c**2))/(gamma_c+1)+\
                         gamma_a*(np.pi/2-2*theta_c)-(np.tan(np.pi/4-theta_c/2)-np.tan(theta_c/2))
    return ke_star   


def hsu_cubes(epsilon, kappa):
    gamma_c = 0.13
    # some work to find gamma_a because it's given as a non-linear function
    # of gamma_c and epsilon.
    # First define the full function
    def F(gamma_a, gamma_c, epsilon):
        return (1-3*gamma_c**2)*gamma_a**3+3*gamma_c**2*gamma_a**2 - (1-epsilon)
    # now i can create function handles so myfunct only takes 1 argument
    myfunct = lambda x: F(x, gamma_c, epsilon)
    import scipy.optimize
    # find gamma_a with an initial guess
    gamma_a = scipy.optimize.broyden1(myfunct, 1., f_tol=1e-14)
    ke_star = 1-gamma_a**2-2*gamma_c*gamma_a+\
              2*gamma_c*gamma_a**2+gamma_c**2*gamma_a**2*kappa+\
              (gamma_a**2-gamma_c**2*gamma_a**2)/(1-gamma_a+gamma_a/kappa)+\
              (2*(gamma_c*gamma_a-gamma_c*gamma_a**2))/(1-gamma_c*gamma_a+gamma_c*gamma_a/kappa)
    return ke_star   

def hsu_zs(epsilon, kappa):
    alpha0 = 0.002
    import scipy.optimize
    def F(B):
        alpha0 = 0.002
        epsilon = 0.42
        return 1-B**2/((1-B)**6*(1+alpha0*B)**2) * \
                ((B**2-4*B+3)+2*(1+alpha0)*(1+alpha0*B)*np.log(((1+alpha0)*B)/(1+alpha0*B)) + \
                    alpha0*(B-1)*(B**2-2*B-1))**2-epsilon
    # find B with an initial guess
    B = scipy.optimize.broyden1(F, 1.5, f_tol=1e-14)
    ke_star = (1-np.sqrt(1-epsilon))+(kappa*np.sqrt(1-epsilon))*(1-1/(1+alpha0*B)**2) + \
            ((2*np.sqrt(1-epsilon))/(1-B/kappa+(1-1/kappa)*alpha0*B))*(((1-1/kappa)*(1+alpha0)*B)/((1-B/kappa+(1-1/kappa)*alpha0*B)**2) * \
                np.log((1+alpha0*B)/((1+alpha0)*B/kappa)) - (B+1+2*alpha0*B)/(2*(1+alpha0*B)**2) - \
                (B-1)/((1-B/kappa+(1-1/kappa)*alpha0*B)*(1+alpha0*B)))
    return ke_star


def helium_mfp(T, P, k_f):
    Rtilde = 8.314 # J/mol-K
    Mg = 4.002602e-3 # kg/mol
    cp = 5.193e3 # J/kg-K
    Mstar = Mg # for helium, monoatomic gas
    # Mstar = 1.4*Mg # for diatomic gas
    mu = 10
    Ts = 600+273
    T0 = 273
    Tr = (Ts-T0)/T0
    a_t = np.exp(-0.57*Tr)*(Mstar/(6.8+Mstar)) + \
          (2.4*mu)/((1+mu)**2)*(1-np.exp(-0.57*Tr))
    l = 2*(2-a_t)/a_t * ((2*np.pi*Rtilde*T)/Mg)**(1/2)*k_f/(P*(2*cp-Rtilde/Mg))
    return l
def zsb(epsilon, kappa, epsilon_r, dp, Tave, P, k_f):
    l = helium_mfp(Tave, P, k_f)
    B = 1.25 * ((1.-epsilon)/epsilon)**(1.055)
    sigma = 5.67e-8
    kappa_r = 4*sigma/(2/epsilon_r-1)*Tave**3*dp/k_f
    kappa_g = 1./(1.+(l/dp))
    phi = 0.01
    N = (1./kappa_g)*(1.+((kappa_r-B)*kappa_g)/kappa)-B*(1./kappa_g-1.)*(1.+kappa_r/kappa)
    #print(N[31:34])
    kc = 2/N*(B*(kappa+kappa_r-1)/(N**2.*kappa_g*kappa)*np.log((kappa+kappa_r)/(B*(kappa_g+(1-kappa_g)*(kappa+kappa_r))))
         +(B+1)/(2*B)*(kappa_r/kappa_g-B*(1+(1-kappa_g)/kappa_g*kappa_r))-(B-1)/(N*kappa_g))
    x = (B*kappa+kappa_r-1)
    y = (N**2*kappa_g*kappa)
    #print(np.argmin(y))
    #plt.plot(kappa,y)
    #plt.plot(kappa,N)
    ke_star = (1-np.sqrt(1-epsilon))*epsilon*((epsilon-1+1/kappa_g)**(-1)+kappa_r) + \
              np.sqrt(1-epsilon)*(phi*kappa+(1-phi)*kc)
    return ke_star




kappa = np.logspace(-1,5,100)


fig = plt.figure(num=0, facecolor='w', edgecolor='k')
ax = fig.gca()
ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$\kappa$')
plt.ylabel(r"$k_{eff}/k_g$")
plt.xlim([min(kappa),max(kappa)])



epsilon = 0.36

k_parallel = epsilon + (1-epsilon)*kappa
k_series = 1/ ( epsilon + (1-epsilon)/kappa )
k_zs = zs(epsilon, kappa)
k_hsu_sc = hsu_sc(epsilon, kappa)
k_hsu_cc = hsu_cc(epsilon, kappa)
k_hsu_cubes = hsu_cubes(epsilon, kappa)
k_hsu_zs = hsu_zs(epsilon, kappa)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# material properties for ZSB (with radiation)
k_s = 2.7
k_f = 0.34 # average He k over 300 to 900 C
dp = 1./1000 # m
Tave = 600+273 # K
epsilon_r = 0.8
P = 0.1013e6 # Pa (1 atm)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
k_zsb = zsb(epsilon, kappa, epsilon_r, dp, Tave, P, k_f)


# DIGITIZED DATA OF VAN ANTWERPEN
kappa_dig, k_e_dig = load_digitized_csv('van-antwerpen-digitzed.csv')

# COMET DATA
k_exp_graphite = [1.251, 1.462, 1.559, 1.629, 1.686, 1.744]
T_exp_graphite = [135.6, 249, 339.1, 417, 484, 544]
k_gas_exp = 0.0025*(np.asarray(T_exp_graphite)+273)**0.72
k_s_exp = k_graphite(T_exp_graphite)
kappa_exp = k_s_exp/k_gas_exp
k_star = k_exp_graphite/k_gas_exp

# CFD-DEM RESULTS
keff_cfd_dem = [ 1.02260048,  0.91373912,  0.80402782,  0.69316134,  0.63713228,  0.58052727,  0.52310535,  0.47031912]
kf_cfd_dem = 0.34
kappa_cfd_dem = [1, .8, .6, .4, .3, .2, .1,.01]
kappa_cfd_dem = np.asarray(kappa_cfd_dem)*2.4/kf_cfd_dem
print(kappa_cfd_dem)
keffstar_cfd_dem = np.asarray(keff_cfd_dem)/kf_cfd_dem

ax.plot(kappa, k_parallel, color=color_idx[0], label='Parallel', linewidth=2)
ax.plot(kappa, k_series, color=color_idx[1], label='Series', linewidth=2)
ax.plot(kappa, k_zs, label='Zehner-Schlunder', color=color_idx[3], linewidth=2) # python 2 can't handle the ascii encoding of his name
ax.plot(kappa, k_hsu_sc, label='Hsu et al, Sq. Cyl.', color=color_idx[4], linewidth=2)
ax.plot(kappa, k_hsu_cc, label='Hsu et al, Circ. Cyl.', color=color_idx[5], linewidth=2)
ax.plot(kappa, k_hsu_cubes, label='Hsu et al, Cubic', color=color_idx[6], linewidth=2)
ax.plot(kappa, k_hsu_zs, label='Hsu et al, ZS corr.', color=color_idx[7], linewidth=2)
#ax.plot(kappa, k_zsb, label='Zehner-Bauer-Schlunder', color=color_idx[8], linewidth=2) # python 2 can't handle the ascii encoding of his name



ax.scatter(kappa_dig, k_e_dig, facecolors=color_idx[0], s=20, zorder=9, label='Experimental data')
# ax.scatter(kappa_exp, k_star, label='COMET data', marker='o', zorder=10, edgecolors='k', facecolors='none', s=20)
ax.scatter(kappa_cfd_dem, keffstar_cfd_dem, facecolors = color_idx[1], zorder=10, s = 20, label='CFD-DEM')

plt.legend(loc='upper left')
plt.savefig('keff-kappa-irradiated', bbox_inches='tight')

plt.show()