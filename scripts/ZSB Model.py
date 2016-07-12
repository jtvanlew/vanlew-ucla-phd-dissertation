
# coding: utf-8

# In[41]:

from numpy import *
from matplotlib.pyplot import *



# In[52]:

def lambda_radiation(epsilon, epsilon_r, T, d, k_solid):
    T = T + 273
    sigma = 5.67E-8 # W m−2 K−4
    B = 1.25 * ((1.-epsilon)/epsilon)**(10./9)
    Lambda = k_solid / (4* sigma * T**3 * d)
    lambda_e1 = (1-sqrt(1-epsilon))*epsilon
    lambda_e2 = (sqrt(1-epsilon))/(2./epsilon_r - 1.)
    lambda_e3 = (B+1.)/B
    lambda_e4 = (1/(1+1/((2./epsilon_r - 1.)*Lambda)))
    lambda_e = (lambda_e1 + lambda_e2*lambda_e3*lambda_e4)*4.*sigma*T**3*d
    return lambda_e

def lambda_fluid(epsilon, lambda_r):
    B = 1.25 * ((1-epsilon)/epsilon)**(10./9)
    lambda_f1 = (1-sqrt(1-epsilon))
    lambda_f2 = (2*sqrt(1-epsilon))/(1-lambda_r*B)
    lambda_f3 = (((1-lambda_r)*B)/(1-lambda_r*B)**2)*log(1./(lambda_r*B)) - (B+1.)/2. - (B-1.)/(1.-lambda_r*B)
    lambda_f = (lambda_f1 + lambda_f2*lambda_f3)
    return lambda_f

def lambda_contact(d, p):
    R = d/2     
    # these are for SC, change for random?
    S = 1
    S_F = 1
    N_A = 1/(4.*(d/2)**2.)
    N_L = 1/(2.*(d/2))
    #------------------
    
    E_s = 120.e9
    nu_s = 0.24
    f_s = p*S_F/N_A
    #solid
    lambda_s1 = ((3*(1-nu_s**2))/(4*E_s) * f_s*R)**(1./3)
    lambda_s2 = (1./(0.531*S))*(N_A/N_L)
    lambda_s = lambda_s1*lambda_s2
    return lambda_s

T = linspace(100,800,100)
k_gas = 0.0025 * (T+273)**0.72 # W/mK

d = 0.001 #m

T_fit = np.array([25,100,200,300,400,500,1000])
k_s_fit = np.array([110,100,92,85,79,75,50])
z = np.polyfit(T_fit, k_s_fit, 1)
k_s = np.poly1d(z)
k_solid = k_s(T)

lambda_r = k_gas / k_solid
kappa = 1/lambda_r

k_exp_graphite = [1.255, 1.544, 1.675, 1.5382, 1.624, 1.676, 1.781]
T_exp_graphite = [140.95, 217.76, 314.85, 390.93, 471.01, 538.42, 594.76]

epsilon = 0.38
epsilon_r = 0.8

k_radiation = lambda_radiation(epsilon, epsilon_r, T, d, k_solid)
k_fluid     = lambda_fluid(epsilon, lambda_r)*k_gas
k_contact   = lambda_contact(d, 1.e2)*k_gas

k_tot = k_radiation + k_fluid + k_contact
k_tot_graphite = k_tot
figure(1)
# plot(T, k_gas, label='Helium')
# plot(T, k_solid, label="Graphite")
plot(T, k_tot,label='total', linewidth=2)
plot(T, k_radiation,label='radiation', linewidth=2)
plot(T, k_fluid,label='fluid + solid', linewidth=2)
plot(T, k_contact,label='contact', linewidth=2)
xlim([min(T),max(T)])
ylim([0, 10])
xlabel('Degrees C')
ylabel(r"$k_{eff}$ (W/m-K)")
legend(loc='best')
grid('on')



show()