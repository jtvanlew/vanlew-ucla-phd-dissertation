import 	numpy 				as 		np 
from 	scipy.optimize 		import 	fsolve
import 	matplotlib.pyplot 	as 		plt

plt.close('all')

n = 3
color = plt.cm.jet(np.linspace(0,1,n))


phis = np.linspace(0.3, 0.64, 100)
Res = np.linspace(0, 50, 6)

F_kc = np.zeros(len(phis))
F_e = np.zeros(len(phis))
F_khl = np.zeros(len(phis))

for j, Re in enumerate(Res):
	for i, phi in enumerate(phis):
		ep = 1. - phi

		#kozeny-carman
		F_kc[i] = 10. *(phi/ep**3.)

		#ergun
		F_e[i] = 8.33 *(phi/ep**3) + 0.18 * Re/ep**3


		# koch-hill-ladd
		if phi < 0.4:
			F0 = 1+3.*np.sqrt(phi/2.)+125./64*phi*np.log(phi)+16.14*phi/(1+0.681*phi-8.48*phi**2+8.16*phi**3)
		else:
			F0 = 10*phi/ep**3
		
		F3 = 0.0673 + 0.212*phi + 0.0232/(ep**5)
		
		F_khl[i] = F0 + F3*Re

	plt.figure(num=j, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')
	plt.xlabel(r'Packing fraction, $\phi$')
	plt.ylabel(r'Dimensionless drag, $F$')
	plt.grid()
	plt.plot(phis, F_kc, label = 'Kozeny-Carman')#, color=color[0])
	plt.plot(phis, F_e, label = 'Ergun')#, color=color[1])
	plt.plot(phis, F_khl, label = 'Koch-Hill-Ladd')#, color=color[2])
	# plt.ylim([0, 1])
	# plt.xlim([0, 1])
	plt.title('Reynolds number = %s'%(np.round(Re,1)))
	plt.legend(loc='best')
	
plt.show()