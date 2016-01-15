import numpy as np
from matplotlib import colors, ticker, cm
import matplotlib.pyplot as plt


T_ = [400,
450,
500,
550,
600,
650,
700,
750,
800,
850,
900,
]

rho_ = [0.071503,
0.06656,
0.062257,
0.058476,
0.055128,
0.052142,
0.049464,
0.047047,
0.044855,
0.042858,
0.041032,
]

mu_ = [3.49E-05,
3.67E-05,
3.85E-05,
4.02E-05,
4.19E-05,
4.36E-05,
4.53E-05,
4.69E-05,
4.85E-05,
5.01E-05,
5.17E-05,
]

cs_ = [1526.8,
1582.5,
1636.3,
1688.3,
1738.9,
1787.9,
1835.7,
1882.3,
1927.7,
1972.1,
2015.5,
]
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

rho_poly_fit_coeffs = np.polyfit(T_, rho_, 2)
mu_poly_fit_coeffs = np.polyfit(T_, mu_, 1)
cs_poly_fit_coeffs = np.polyfit(T_, cs_, 1)

mu = np.poly1d(mu_poly_fit_coeffs)
rho = np.poly1d(rho_poly_fit_coeffs)
cs = np.poly1d(cs_poly_fit_coeffs)

T = [400, 800, 900]

M = 6.6464764063e-27			# Molecular mass, kg
kb = 1.3806504e-23				# Boltzmann constant, J/K
p = 0.1e6	 					# Ambient helium pressure, Pa
L = np.logspace(-8, -2, 100) 	# mean pore size, m 
d = 28.e-12						# particle hard shell diameter, m

fig, ax = plt.subplots()
for j, T in enumerate(T):
	mfp = mu(T)/rho(T)*np.sqrt(np.pi*M/(2*kb*(T+273)))
	Re = rho(T)*(0.05)*L/mu(T)
	Ma = 0.05/cs(T)
	gamma = 5/3.

	Kn1 = mfp/L
	Kn3 = Ma/Re * (np.sqrt(gamma*np.pi/2))
	
	print('Length scale above which Kn < 10^-2 is satisfied')
	print(L[np.where(Kn1<0.01)][0])
	print(T, mfp)
	ax.loglog(L, Kn1, label = '%s C'%(T), linewidth = 2, color = color_idx[j])
	#ax.loglog(L, Kn3, label = 'Kn3, '+str(T), linestyle = '--', linewidth = 2, color = color_idx[j+2])

print(mu(0)/rho(0)*np.sqrt(np.pi*M/(2*kb*(0+273)))*10**9)
plt.legend(loc='best')
plt.grid('on')
plt.xlabel("Characteristic Length (m)")
plt.ylabel("Knudsen Number")
plt.show()