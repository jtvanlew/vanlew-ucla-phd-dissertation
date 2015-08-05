import numpy as np
ep = 0.2 # void fraction of particle
delta = .07 #enrichment of Li-6 
T = np.linspace(700,1100,100)

nu = 0.3*(1-ep)

V = 363*(1-2.36*ep)

rho = 3.44*(1-1.82e-2*delta)

cp = 355*(T-100)**1.1/(1+0.3*T**1.05)
k = (1-ep)**2.9*(5.35-4.78e-3*T+2.87e-6*T**2)

#beta = 1.154e-5 + 1.101e-8*T
beta = (-0.004119 + 1.154e-5*T +5.505e-9*T**2)/(T-293)