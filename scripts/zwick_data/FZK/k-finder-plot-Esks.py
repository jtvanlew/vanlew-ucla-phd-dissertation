import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
#import pylab as P
import matplotlib.cm as mplcm
import matplotlib.colors as colors

import scipy.stats as ss
import scipy as sp

Estand = 220.e9
nustand = 0.27
Fn0 = 0.1
nupeb = 0.24
Epebbulk = 90.e9


if len(sys.argv)>2:
	filenames = sys.argv[1:]
else:
	filenames = [sys.argv[1]]


endpoint = 0
j = 0
dp = np.zeros(len(filenames))
Fmax = np.zeros(len(filenames))
W = np.zeros(len(filenames))
Fpoly = np.zeros((len(filenames),3))
kpeb = np.zeros(len(filenames))
err_peb = np.zeros(len(filenames))
E_rec = [[] for i in range(len(filenames))]
for loadname in filenames:
	# Open the file again to read the pebble diameter
	print 'Finding diameter and strain energy for: '+loadname
	f = open(loadname)
	print loadname
	content = f.readlines()
	dp1 = content[1]
	w1 = content[2]
	dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.
	W[j] = float(w1[16:(w1.find("mJ")-2)])
	j +=1

W = W/1000.
fit_alpha,fit_loc,fit_beta=ss.gamma.fit(W)
plt.figure(5)
Wfit = ss.gamma.rvs(fit_alpha,loc=fit_loc,scale=fit_beta,size=len(W))
print fit_alpha, fit_loc, fit_beta

plt.hist(Wfit)
plt.hist(W)
j = 0



k = np.linspace(0.01,1,1000)

for loadname in filenames:
	print 'Calculating elasticity reduction factor for: '+loadname

	filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
	s = filedata[:,0]
	F = filedata[:,1]
	
	# Search for the maximum force, discard the rest of the plot
	# after this point
	maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
	F = F[0:maxIndex]	
	s = s[0:maxIndex]

	# Make a polynomial fit
	sfit = np.linspace(0,s[-1],1001)
	Fpoly = np.poly1d(np.polyfit(s,F,3, full=False))
	Ffit = Fpoly(sfit)

	# Find crush force
	Fmax[j] = F[-1]


	# endpoint for the plot
	if s[-1] > endpoint:
		endpoint = s[-1]


	err = 1e5
	for i in k:
		# Using Hertz theory, find the predicted force for 
		# given displacement
		Epeb = i*Epebbulk
		Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
		# in the experiment, there is overlap causing the pre-load
		# use that preload to back-out the initial overlap
		so = ((3*Fn0/Estar)**2*(1/dp[j]))**(1./3)

		# Hertz force (with corrected initial strain)
		Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+sfit/1000.)**3)+Fn0
		diff = Fhertz - Ffit
		
		erri = np.linalg.norm(diff)
		
		if erri < err:
			err = erri
			kpeb[j] = i
			err_peb[j] = erri

	Epeb = kpeb[j]*Epebbulk
	E_rec[j] = "E = "+ str(Epeb/10**9) + " GPa"
	Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
	Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+s/1000.)**3)+Fn0


	j+=1


plt.figure(1)
plt.plot(np.sort(kpeb*Epebbulk)/10.**9, 'o')
plt.xlabel('Sorted pebble number')
plt.ylabel("Modified Young's modulus (GPa)")


import scipy
R = scipy.stats.pearsonr(dp, kpeb)

print R

plt.figure(101)
plt.scatter(dp*1000, kpeb, s=60, color='g')
plt.xlim(min(dp*1000), max(dp*1000))
plt.ylabel('Elasticity reduction factor')
plt.xlabel('Pebble diameter (mm)')
if len(filenames)>1:
	plt.figure(102)

	n, bins, patches = plt.hist(kpeb, histtype='stepfilled')
	plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
	plt.xlabel('Elasticity reduction factor')
	plt.ylabel('Count')

plt.show()