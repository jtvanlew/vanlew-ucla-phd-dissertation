# This file wants scatter data from a .txt file that has been created by Ovito
import numpy as np
import matplotlib.pyplot as plt
import sys, os

color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

#loadfile1 = input('load file 1 name...')
loadfiles = ['9','8','7','6','5','4','3','2','1','0.1']

labels = [r'$\eta = 0.9$',
			r'$\eta = 0.8$',
			r'$\eta = 0.7$',
			r'$\eta = 0.6$',
			r'$\eta = 0.5$',
			r'$\eta = 0.4$',
			r'$\eta = 0.3$',
			r'$\eta = 0.2$',
			r'$\eta = 0.1$',
			r'$\eta = 0.01$']

#fig, ax = plt.subplots(1)
fig, ax = plt.subplots()

alpha = 0.2
ax.set_xlabel('Dimensionless x')
ax.set_ylabel(r'Temperature (K)')
ax.set_xlim([-1, 1])
ax.grid("on")
ax.set_ylim([573, 4000])
n = np.size(loadfiles)


k = np.zeros(len(loadfiles))
Tmid = np.zeros(len(loadfiles))
phi = np.zeros(len(loadfiles))

for j, loadfile in enumerate(loadfiles):
	data = np.loadtxt(loadfile)

	for row in data:
		x = data[:,0]/0.01
		temperatures = data[:,1]

	if j<8:
		ax.scatter(x,temperatures,
				color=color_idx[j], alpha=alpha)


	xspan = np.abs(np.max(x)) + np.abs(np.min(x))
	Nbins = 20
	tempbin = np.zeros([Nbins+2])
	xbin = np.zeros([Nbins+2])

	xbin[0], xbin[-1] = -1,1
	tempbin[0], tempbin[-1] = 573,573
	dx = xspan/Nbins

	x0 = np.min(x)
	for i in np.arange(1,Nbins+1):
		x1 = x0 + dx
		xbin[i] = x0 + dx/2.
		tempbin[i] = 0
		count = 0
		index = 0
		for xvalue in x:
			if xvalue >= x0 and xvalue < x1:
				tempbin[i] += temperatures[index]
				count += 1
			index += 1
		tempbin[i] /= count
		x0 = x1

	Ti = tempbin[len(tempbin)/2]
	Tmid[j] = Ti
	To = 573
	natoms = len(data)
	qn = 4./3 * np.pi * (0.0005)**3
	k[j] = (8.e6/.64)*0.64*(0.01**2)/(2*(Ti-To))

	k[j] = k[j]

	
	if j < 8:
		ax.plot(xbin, tempbin, color = color_idx[j],label = labels[j], linewidth=2 )



#plt.title(r'$k_{eff}$ = %s'%np.round(k0,3)+ " W/m-K")
#plt.legend(loc='best')


# epsFile = 'dem-evap-0-15-scatter-keff.eps'
# plt.savefig(epsFile)
plt.legend(loc='best')
pngFile = 'irradiated-temperatures.png'
plt.savefig(pngFile)

array = [1, .90, .80, .70, .60, .50, .40, .30, 0.2, .1, .01]
k = np.insert(k, 0, 0.398)
k = k/k[0]

plt.figure(2)
plt.scatter(array, k, linewidth = 2, color = color_idx[0], label= "DEM data")
plt.grid('on')
plt.xlabel(r'$k_{irr}/k_{unirr}$')
plt.ylabel(r'$k_{eff}/k_{eff,unirr}$')
plt.ylim([0, 1])


z = np.polyfit(array, k, 5)
p = np.poly1d(z)
print(p) 	
# print(p)
xx = np.linspace(0, 1, 100)
# found the data to fit a power law from excel
#p = 0.9878*xx**0.6558
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(array, k)
# print(r_value)
# print(slope)
plt.plot(xx, p(xx), linewidth = 1, color = color_idx[1], label = "DEM fit")

pngFile = 'keff-plots.png'
plt.legend(loc='best')
plt.xlim([0, 1.00])
plt.savefig(pngFile)



plt.figure(3)
Tmid = np.insert(Tmid, 0, 1577)
# z = np.polyfit(array, Tmid, 4)
# p = np.poly1d(z)
# print(p)
# found the data to fit a power law from excel
p = 1536.2*xx**-0.526
plt.scatter(array, Tmid, linewidth = 2, color = color_idx[0], label= "DEM data")
plt.plot(xx, p, linewidth = 1, color = color_idx[1], label = "DEM fit")
plt.legend(loc='best')
plt.xlim([0, 1.00])
plt.grid('on')
plt.xlabel(r'$k_{irr}/k_{unirr}$')
plt.ylabel('Average midline temperature (K)')

pngFile = 'Tmid-plots.png'
plt.savefig(pngFile)

plt.show()