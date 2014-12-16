import numpy as np 

Nu    = 2.
kf    = 0.3
kr    = 2.4           # W/mK 
rhor  = 2260.          # kg/m**3 
Cr    = 1134.          # J/kg*K 
D     = 1./1000        # m

h1    = Nu*kf/(D)    # W/m2K
Bi    = h1*D/(kr)   #
h2    = h1/(1.+Bi/5.)   # W/m2K -- Jeffreson Correction
alfa  = 4e-6      # m2/s


# h1=355.1
# kr=2.8
# rhor=2630
# Cr=775
# D=.04
# Bi=h1*D/(2*kr)
# h2=h1/(1+Bi/5)
# alfa=1.3737e-06


V     = 4./3.*np.pi*(D/2)**3 # m3
A     = 4*np.pi*(D/2)**2  # m2
t     = np.linspace(0,10,300) # s
from scipy.optimize import fsolve
def func(x):
    return 1 - Bi - x * (1./np.tan(x))

zeta = []
for x in np.arange(0,20,0.5):
  zeta_new = np.round(float(fsolve(func, x)),7)
  if zeta_new > 0:
    if zeta_new not in zeta:
      zeta.append(zeta_new)


C = np.array([(4*(np.sin(i)-i*np.cos(i)))/(2*i-np.sin(2*i)) for i in zeta])


Fo = np.zeros(len(t))
Q1 = np.zeros(len(t))
Q2 = np.zeros(len(t))
Q3 = np.zeros(len(t))
theta1 = np.zeros(len(t))
theta2 = np.zeros(len(t))
theta3 = np.zeros(len(t))
for i in np.arange(len(t)):
    # Q1 is Q_out,lumped capacitance
    # Q2 is Q_out,jeffreson correction
    # Q3 is Q_out,transient 

    
    theta1[i]=np.exp(-(h1*A/(rhor*Cr*V))*t[i]);
    Q1[i]=1.-theta1[i];

    theta2[i]=np.exp(-(h2*A/(rhor*Cr*V))*t[i]);
    Q2[i]=1.-theta2[i];
    
    Fo[i]=(alfa/(D/2.)**2)*t[i]
    theta3[i] = 0
    Q3[i]=1
    for j in np.arange(len(zeta)):
      theta3[i] += C[j]*np.exp(-zeta[j]**2*Fo[i])*(1./zeta[j])*np.sin(zeta[j])
      Q3[i] -= 3*(C[j]*np.exp(-zeta[j]**2*Fo[i])*(1./zeta[j]**3)*(np.sin(zeta[j])-zeta[j]*np.cos(zeta[j])))


import matplotlib.pyplot as plt 
plt.figure(1)
plt.xlabel('Fourier Number')
plt.ylabel(r'$Q^*$')

plt.plot(Fo,Q1, label = 'Lumped Capacitance Model')
plt.plot(Fo,Q2, label = 'Jeffreson Correction')
plt.plot(Fo,Q3, label = 'Transient Conduction Solution')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.legend(loc='best')

plt.show()