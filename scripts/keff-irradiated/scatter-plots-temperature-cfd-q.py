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
loadfiles = ['cfd-q.txt','8-cfd-q.txt','6-cfd-q.txt','4-cfd-q.txt','3-cfd-q.txt','2-cfd-q.txt','1-cfd-q.txt', '01-cfd-q.txt']
labels = [r'$\eta=1$',r'$\eta = 0.8$',r'$\eta = 0.6$',r'$\eta = 0.4$',r'$\eta = 0.3$',r'$\eta = 0.2$',r'$\eta = 0.1$',r'$\eta = 0.01$']

fig, ax = plt.subplots()

alpha = 0.2
ax.set_xlabel('Dimensionless x')
ax.set_ylabel(r'Temperature (K)')
ax.set_xlim([-1, 1])
ax.grid("on")
ax.set_ylim([573, 1400])
n = np.size(loadfiles)


k = np.zeros(len(loadfiles))
Tmid = np.zeros(len(loadfiles))
phi = np.zeros(len(loadfiles))


for j, loadfile in enumerate(loadfiles):
	data = np.loadtxt(loadfile)

	for row in data:
		x = data[:,0]/0.01
		temperatures = data[:,1]
	if j>0:
		ax.scatter(x,temperatures,
				color=color_idx[j-1], alpha=alpha)

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
	k[j] = (8.e6)*(0.01**2)/(2*(Ti-To))

	k[j] = k[j]

	if j>0:
		ax.plot(xbin, tempbin, color = color_idx[j-1], label = labels[j], linewidth=2 )






plt.legend(loc='best')
pngFile = 'irradiated-temperatures-cfd-q.png'
plt.savefig(pngFile)



array = [1, .8, .6, .4, .3, .2, .1,.01]
print(k)
# k = np.insert(k, 0, 1.019)
k = k/k[0]
plt.figure(2)
plt.scatter(array, k, linewidth = 2, color = color_idx[0], label= "DEM data")
plt.grid('on')
plt.xlabel(r'$k_{irr}/k_{unirr}$')
plt.ylabel(r'$k_{eff}/k_{eff,unirr}$')



z = np.polyfit(array, k, 1)
p = np.poly1d(z)
print(p)
xx = np.linspace(0, 1, 100)
plt.plot(xx, p(xx), linewidth = 1, color = color_idx[1], label = "DEM fit")

pngFile = 'keff-plots-cfd-q.png'
plt.legend(loc='best')
plt.xlim([0, 1.00])
plt.ylim([0, 1])
plt.savefig(pngFile)



plt.figure(3)
# Tmid = np.insert(Tmid, 0, 823)
# print((Tmid-573))
# Tmid = Tmid/Tmid[0]
z = np.polyfit(array, Tmid, 3)
p = np.poly1d(z)
print(p)
plt.scatter(array, Tmid, linewidth = 2, color = color_idx[0], label= "DEM data")
plt.plot(xx, p(xx), linewidth = 1, color = color_idx[1], label = "DEM fit")
plt.legend(loc='best')
plt.xlim([0, 1.00])
# plt.ylim([1, 1.5])
plt.grid('on')
plt.xlabel(r'$k_{irr}/k_{unirr}$')
plt.ylabel('Average midline temperature (K)')

pngFile = 'Tmid-plots-cfd-q.png'
plt.savefig(pngFile)

plt.show()