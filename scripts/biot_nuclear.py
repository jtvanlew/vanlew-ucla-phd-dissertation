import 	numpy 				as 		np 
from 	scipy.optimize 		import 	fsolve
import 	matplotlib.pyplot 	as 		plt

plt.close('all')
plt.figure(num=1, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel(r'Dimensionless time, $\tau$')
plt.ylabel(r'Dimensionless energy, $E^*$')
plt.grid()

plt.figure(num=2, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel(r'Dimensionless time, $\tau$')
plt.ylabel(r'Error')
plt.grid()

def error(x,y):
	return np.abs(x-y)/y


# SOLID MATERIAL PROPERTIES##########################
D     = 1./1000        			# [m]
V     = 4./3.*np.pi*(D/2)**3 	# [m3]
A     = 4*np.pi*(D/2)**2  		# [m2]
kr    = 2.4		      			# [W/mK ]
rhor  = 2260.          			# [kg/m**3]
Cr    = 1134.          			# [J/kg*K]
alfa  = kr/(rhor*Cr)      		# [m2/s]
tau   = (D/2)**2/alfa     		# [s] - time constant
q     = 8.e6  # W/m3
#####################################################


# FLUID MATERIAL PROPERTIES##########################
Nu    = 2.
kf    = 0.29445875
h1    = Nu * kf * 2 / D
#####################################################



# SOLID-FLUID INTERACTION PROPERTIES#################
Bi    = np.round(h1 * D / (2 * kr),3)	# Biot number
Bip   = Bi / (1.+Bi/5.) 
# T0 is initial temperature difference between solid-fluid (T0 - Tf)
T0 	  = 20.  				# [K]
#####################################################


tauscale=3
# SYSTEM PROPERTIES #################################
t     = np.linspace(0,tauscale*tau/Bi,100) # [s]
#####################################################




# DIMENSIONLESS VALUES###############################
G     = q * (D/2.)**2 / (kr*T0) 	# heat generation
tstar    = t/tau 						# time
#####################################################



# SOLVE TRANSCENDENTAL EQUATION #####################
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
######################################################



# CALCULATE ALL ZETA-DEPENDENT VALUES ################
zeta = np.array([i for i in zeta])
C = np.array([(np.sin(i) - i*np.cos(i))/i**2 for i in zeta])
K = np.array([(3.*(i**2-2.)*np.sin(i) - i*(i**2-6.)*np.cos(i))/i**4 for i in zeta])
Z = (1 - (G/6.)*(1+2./Bi))*C + (G/6.)*K
N = np.array([(1./2) * (i**2 + (Bi-1)**2 + (Bi-1))/(i**2 + (Bi-1)**2) for i in zeta])
######################################################



######################################################
# FIND DIMENSIONLESS TEMPERATURES AND SPHERE ENERGIES
# E1 is lumped capacitance
# E2 is jeffreson correction
# E3 is transient w/ thermal gradient
theta1  = (1.-G/(3*Bi))*np.exp(-3*Bi*tstar) + G/(3*Bi)
theta2  = (1.-G/(3*Bip))*np.exp(-3*Bip*tstar) + G/(3*Bip)
E1      = theta1
E2      = theta2
E3      = np.zeros(len(t)) + G*(1./15 + 1./(3*Bi))
for j in np.arange(len(zeta)):
  E3 += 3*np.exp(-zeta[j]**2*tstar)*Z[j]*C[j]/N[j]
err = error(E1,E3)
errJ = error(E2,E3)
######################################################




######################################################
# PLOT JUNK
plt.figure(1)
plt.title('Bi = %s'%(Bi))
plt.plot(tstar, E1, marker = 'x', label = 'LC', color='k')
# plt.plot(tstar, E2, marker = 'o', label = 'Jeffreson', color='k')
plt.plot(tstar, E3, label = 'Exact', color='k')
plt.ylim([0, 1])
plt.xlim([0, tauscale/Bi])
plt.legend(loc='best')

plt.figure(2)
plt.title('Bi = %s'%(Bi))
plt.xlim([0, tauscale/Bi])
plt.semilogy(tstar, err, 'k',label='LC error')
plt.semilogy(tstar, errJ, 'k--', label='Jeffreson error')
plt.legend(loc='best')
######################################################


plt.show()

