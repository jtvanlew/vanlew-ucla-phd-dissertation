import 	numpy 				as 		np 
from 	scipy.optimize 		import 	fsolve
import 	matplotlib.pyplot 	as 		plt
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [207./255, 207./255, 207./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255],
             [148./255, 103./255, 189./255],
             [95./255, 158./255, 209./255],
             [44./255, 160./255, 44./255],
             [152./255, 223./255, 138./255]
             ]

def wakao_kaguei(Re,Pr):
	return 2+1.1*Re**(0.6)*Pr**(1/3.)

def ranz_marshall(Re, Pr):
	return 2. + 0.6*Pr**(1/3.)*Re**(1/2.)

def achenbach(Re, Pr):
	return 2.0 + (0.25*Re + 3.e-4*Re**(1.6) )

def gnielinski(Re, Pr):
	Nul = 0.664*Re**(1/2.)*Pr**(1./3)
	Nut = (0.037*Re**(0.8)*Pr)/(1.+2.443*Re**(-0.1)*Pr**(2./3)-1.)
	return 2 + np.sqrt(Nul**2 + Nut**2)

def ranz_marshall_m(Re, Pr, epsilon):
	return 2 + 0.6*epsilon**3.5*Re**(1./2)*Pr**(1./3)

def kemp(Re, Pr):
	return 2 + 0.5*epsilon**3.5*Re**(1./2)*Pr**(1./3)

def frantz(Re, Pr, epsilon):
	return 2+ 0.000045*epsilon**3.5* Re**(1./2)

def bandrowski(Re, epsilon):
	return 0.00114*epsilon**(-0.5984)*Re**(0.8159)

def whitaker(Re, Pr):
	return 2 + (0.4*Re**(0.5)+0.06*Re**(0.67))*Pr**(0.4)

def gunn(Re, Pr, epsilon):
	return (7.-10.*epsilon+5.*epsilon**2)*(1+0.7*Re**(0.2)*Pr**(1/3))+(1.33-2.4*epsilon+1.2*epsilon**2)*Re**(0.7)*Pr**(1/3)

def zhou(Re, Pr,):
	return 2 + 0.2*Re**(1./2)*Pr**(1./3)

epsilon = 0.36
Pr = 0.7

Re = np.logspace(0, 2, 100)

fig = plt.figure(num=0, figsize=(12, 9), dpi=150, facecolor='w', edgecolor='k')
ax = fig.gca()
ax.grid(True)

ax.set_xscale('log')
plt.xlabel(r'Reynolds number')
plt.ylabel(r"Nusselt number")


ax.plot(Re, wakao_kaguei(Re, Pr) , color=color_idx[0], label='Wakao & Kaguei', linewidth=2)
ax.plot(Re, achenbach(Re, Pr), color=color_idx[1], label='Achenbach', linewidth=2)
ax.plot(Re, gnielinski(Re, Pr), color=color_idx[2], label='Gnielinski', linewidth=2)
ax.plot(Re, ranz_marshall_m(Re, Pr, epsilon), color=color_idx[3], label='Ranz & Marshall', linewidth=2)
ax.plot(Re, ranz_marshall(Re, Pr), color=color_idx[4], label='modified Ranz & Marshall', linewidth=2)
ax.plot(Re, kemp(Re, Pr), color=color_idx[5], label='modified Kemp', linewidth=2)
ax.plot(Re, frantz(Re, Pr, epsilon), color=color_idx[6], label='modified Frantz', linewidth=2)
ax.plot(Re, bandrowski(Re, epsilon), color=color_idx[7], label='Bandrowski', linewidth=2)
ax.plot(Re, whitaker(Re, Pr), color=color_idx[8], label='Whitaker', linewidth=2)
ax.plot(Re, gunn(Re, Pr, epsilon), color=color_idx[9], label='Gunn', linewidth=2)
ax.plot(Re, zhou(Re, Pr), color=color_idx[10], label='Zhou', linewidth=2)


plt.legend(loc='best')
	
plt.show()