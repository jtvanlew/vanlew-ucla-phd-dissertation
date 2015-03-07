import 	numpy 				as 		np 
from 	scipy.optimize 		import 	fsolve
import 	matplotlib.pyplot 	as 		plt

plt.close('all')

n = 3
color = plt.cm.jet(np.linspace(0,1,n))


phis = np.linspace(0.2, 0.64, 100)
Res = np.linspace(0, 50, 6)

Nu_w = np.zeros(len(phis))
Nu_lm = np.zeros(len(phis))

Pr = 0.7

for j, Re in enumerate(Res):
	for i, phi in enumerate(phis):
		ep = 1. - phi

		# wakao
		Nu_w[i] = 2 + 1.1 * Re**(0.6)*Pr**(1./3)

		#ergun
		Nu_lm[i] = 2 + 0.6*ep**3.5*Re**(1/2.)*Pr**(1./3)

	plt.figure(num=j, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')
	plt.xlabel(r'Packing fraction, $\phi$')
	plt.ylabel('Nusselt number, Nu')
	plt.grid()
	plt.plot(phis, Nu_lm, label = 'Li & Mason')#, color=color[0])
	plt.plot(phis, Nu_w, label = 'Wakao')#, color=color[1])
	# plt.ylim([0, 1])
	# plt.xlim([0, 1])
	plt.title('Reynolds number = %s'%(np.round(Re,1)))
	plt.legend(loc='best')
	
plt.show()