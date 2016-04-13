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
loadfiles = ['8-cfd.txt','6-cfd.txt','4-cfd.txt','3-cfd.txt','2-cfd.txt','1-cfd.txt','0.1-cfd.txt']


n = np.size(loadfiles)
k = np.zeros(len(loadfiles))
Tmid = np.zeros(len(loadfiles))
phi = np.zeros(len(loadfiles))


for j, loadfile in enumerate(loadfiles):
	data = np.loadtxt(loadfile)
	for row in data:
		x = data[:,0]/0.01
		temperatures = data[:,1]

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
	k[j] = (8.e6)*0.64*(0.01**2)/(2*(Ti-To))




array = [1, .8, .6, .4, .3, .2, .1, .01]
Tmid = np.insert(Tmid, 0, 823)
Tstar = (Tmid - To)/(Tmid[0]-To)
#Tstar = Tmid/Tmid[0]
plt.scatter(array, Tstar, s=60, linewidth = 2, edgecolors = color_idx[0], facecolors='None', label= r"Q = %s MW/m$^3$"%(8*.64))


































loadfiles = ['cfd-q.txt','8-cfd-q.txt','6-cfd-q.txt','4-cfd-q.txt','3-cfd-q.txt','2-cfd-q.txt','1-cfd-q.txt', '01-cfd-q.txt']


n = np.size(loadfiles)
k = np.zeros(len(loadfiles))
Tmid = np.zeros(len(loadfiles))
phi = np.zeros(len(loadfiles))


for j, loadfile in enumerate(loadfiles):
	data = np.loadtxt(loadfile)

	for row in data:
		x = data[:,0]/0.01
		temperatures = data[:,1]

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








array = [1, .8, .6, .4, .3, .2, .1, .01]

Tstar = (Tmid - To)/(Tmid[0]-To)

#Tstar = Tmid/Tmid[0]

plt.scatter(array, Tstar, s=20, marker = 's', linewidth=2, edgecolors= color_idx[1], facecolors='None', label= r"Q = %s MW/m$^3$"%(8.00))

plt.xlim([0, 1.00])
# plt.ylim([1, 1.5])
plt.grid('on')
plt.xlabel(r'$k_{irr}/k_{unirr}$')
plt.ylabel(r'Normalized bed temperature, $\Theta$')

z = np.polyfit(array, Tstar, 3)
p = np.poly1d(z)
print(p)
xx = np.linspace(0, 1, 100)
plt.plot(xx, p(xx), linewidth = 1, color = color_idx[3], label = "Third-order fit")

plt.ylim([1, 2.4])
plt.legend(loc='best')
pngFile = 'Tmid-plots-cfd-Tmid-comparison.png'
plt.savefig(pngFile)



plt.show()