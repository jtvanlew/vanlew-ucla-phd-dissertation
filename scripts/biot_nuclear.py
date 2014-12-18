import numpy as np 

D     = 1./1000        # m
V     = 4./3.*np.pi*(D/2)**3 # m3
A     = 4*np.pi*(D/2)**2  # m2

kr    = 2.4           # W/mK 
rhor  = 2260.          # kg/m**3 
Cr    = 1134.          # J/kg*K 


alfa  = kr/(rhor*Cr)      # m2/s

Nu    = 2
kf    = 0.29445875
h1    = Nu * kf * 2 / D
Bi    = h1 * D / (2 * kr)
h2    = h1/(1.+Bi/5.)   # W/m2K -- Jeffreson Correction

deltaT = 50.
q     = 8.e6
f     = q / (rhor*Cr*deltaT)
#G     = q * (D/2.)**2 / (kr)
theta0 = 0
tau   = (D/2)**2/alfa     # s

t     = np.linspace(0,3*tau,100) # 1/s
Fo    = t/tau





from scipy.optimize import fsolve

def func(x):
    return 1 - x * (1./np.tan(x)) - Bi
  
zeta = []
x = 0.0001
Nroots = 20

while len(zeta) < Nroots:
  zeta_new = np.round(float(fsolve(func, x)),4)
  if np.abs(zeta_new) not in zeta:
    zeta.append(np.abs(zeta_new))
  x += 0.5

zeta = np.array([i for i in zeta])
C = np.array([(np.sin(i) - i*np.cos(i))/i**2 for i in zeta])
K = np.array([(3.*(i**2-2.)*np.sin(i) - i*(i**2-6.)*np.cos(i))/i**4 for i in zeta])
Z = (theta0 - (1./6.)*(1+2./Bi))*C + K/6.
N = np.array([(1./2) * (i**2 + (Bi-1)**2 + (Bi-1))/(i**2 + (Bi-1)**2) for i in zeta])


#C = np.array([(4*(np.sin(i)-i*np.cos(i)))/(2*i-np.sin(2*i)) for i in zeta])






# Q1 is Q_out,lumped capacitance
# Q2 is Q_out,jeffreson correction
# Q3 is Q_out,transient w/ thermal gradient


theta1n  = np.exp(-(h1*A/(rhor*Cr*V))*t) + f*t
theta1  = np.exp(-(h1*A/(rhor*Cr*V))*t)
theta2  = np.exp(-(h2*A/(rhor*Cr*V))*t) + f*t

Q1      = 1.-theta1
Q2      = 1.-theta2

r = np.linspace(0,1,20)
Q3     = np.zeros(len(t))
theta3 = np.zeros(len(r))


for j in np.arange(len(zeta)):
  Q3 += 3*(np.exp(-zeta[j]**2*Fo)*Z[j]*C[j]/N[j])

for t in Fo:
  for j in np.arange(len(zeta)):
    Q3 += 3*(np.exp(-zeta[j]**2*Fo)*Z[j]*C[j]/N[j]) 
    


import matplotlib.pyplot as plt 
plt.close('all')
plt.figure(1)
plt.xlabel(r'Dimensionless time ($t/\tau$)')
plt.ylabel(r'Dimensionless total heat removed, $Q^*$')
plt.title(r'Biot Number = %s'%(np.round(Bi,2)))
# plt.plot(Fo, Q1, label = 'Lumped capacitance model (LC)')
# plt.plot(Fo, Q2,  marker = 'o', label = 'LC with Jeffreson correction (JC)')
plt.plot(Fo, Q3, label = 'Exact solution with thermal gradient (TG)')
#plt.ylim([0, 1])
plt.legend(loc='best')


plt.show()