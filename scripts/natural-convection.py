import numpy as np

rho = 0.071503 #kg/m3
g = 9.8 #m/s2
K = 6.328e-10 #m2 calculated from kozeny-carman relation
mu = 4.35e-5 #Pa-s
H = 0.01 #m
dT = 500 #K
Tbar = (900+273+400+273)/2
ks = 2.4 #W/m-K
kf = 0.34 #W/m-K
km = 0.64*ks + 0.36*kf
alphaf = 1.281e-3 #m2/s
alpham = km/kf*alphaf

beta = 1/Tbar
nu = mu/rho
Ra = rho*g*K*H*dT/(Tbar*mu*alpham)

q = 0.1e8
# C = rho*g*H*dT/(Tbar*mu*alpham)
Ra2 = (g*beta*q*H**4)/(nu*alphaf*kf)
print(Ra2)
#ep = np.linspace(0.36,0.9,100)
dp = np.linspace(0.001, 10000, 10000)
ep = 0.36

#e, d = np.meshgrid(ep, dp)
K = (dp**2)/(180) * (ep**3)/((1-ep)**2)

print(dp[np.where(K>1.99e-3)][0])
# import matplotlib.pyplot as plt
# plt.contour(e, d, K, 1000)
# plt.show()

# dp = 0.001

# ep = np.linspace(.36, .99999, 10000000)
# K = (dp**2)/(180) * (ep**3)/((1-ep)**2)

# print(ep[np.where(K>1.99e-3)][0])

nuf = 8.52e-4
Raregular = g*H**3*dT/(Tbar*nuf*alphaf)
print(Raregular)


kf = 13.12
mu = 0.0022
rho = 9486
cp = 200.22

nu = mu/rho
alpha = kf/(rho*cp)

L = 0.05
q = 0.1e8
beta = 1.77e-4

Ra3 = (g*beta*q*L**4)/(nu*alpha*kf)
print(Ra3)

