import numpy as np
import matplotlib.pyplot as plt
import sys, os
#import pylab as P


if len(sys.argv)>2:
	filenames = sys.argv[1:]
else:
	filenames = [sys.argv[1]]


endpoint = 0
j = 0
dp = np.zeros(len(filenames))
Fmax = np.zeros(len(filenames))
Wstar = np.zeros(len(filenames))
Fpoly = np.zeros((len(filenames),3))


for loadname in filenames:

	filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
	s = filedata[:,0]
	F = filedata[:,1]
	
	# Search for the maximum force, discard the rest of the plot
	# after this point
	maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
	F = F[0:maxIndex]	
	s = s[0:maxIndex]

	# Make a polynomial fit
	# sfit = np.linspace(0,s[-1],1001)
	# Fpoly = np.poly1d(np.polyfit(s,F,3, full=False))
	# Ffit = Fpoly(sfit)

	# Find crush force
	Fmax[j] = F[-1]


	# endpoint for the plot
	if s[-1] > endpoint:
		endpoint = s[-1]
	
	# Integrate Force over displacement for strain energy
	W = np.trapz(F,s)

	# Open the file again to read the pebble diameter
	f = open(loadname)
	content = f.readlines()
	dp1 = content[1]
	dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.


	# Find strain energy per pebble volume
	Wstar[j] = W/(4/3. * np.pi * (dp[j]/(2.))**3.)
	
	# Using Hertz theory, find the predicted force for 
	# given displacement
	Epeb = 120.e9
	nupeb = 0.27

	Estand = 220.e9
	nustand = 0.24

	Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
	# in the experiment, there is overlap causing the pre-load
	# use that preload to back-out the initial overlap
	Fn0 = 0.1
	so = ((3*Fn0/Estar)**2*(1/dp[j]))**(1./3)
	# Hertz force (with corrected initial strain)
	Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+s/1000.)**3)+Fn0
	

	# Plot all forces over displacements
	labelstring = 'Pebble '+str(j)
	labelstringH = "Hertz for pebble "+str(j)
	plt.figure(1)
	plt.plot(s, F)#, label=labelstring)
	#plt.plot(s, Fhertz, '-.')#,label=labelstringH)
	# plt.plot(sfit,Ffit)
	plt.legend(('Experimental',"Hertz theory (E = "+str(Epeb/10.**9)+"GPa )"))
	plt.xlabel('Standard travel (mm)')
	plt.ylabel('Standard force (N)')
	plt.xlim((0, endpoint))
	plt.ylim((0,10))
	#plt.legend(['Experimental','Hertzian'])
	j+=1



# Plot histograms if more than 1 file is loaded
if len(filenames)>1:
	plt.figure(2)

	n, bins, patches = plt.hist(Fmax, 10, normed=1, histtype='stepfilled')
	plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
	plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
	plt.xlabel('Crush force (N)')
	plt.ylabel('Normalized probability')
	
	plt.figure(3)
	n, bins, patches = plt.hist(Wstar, 10, normed=1, histtype='stepfilled')
	plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
	plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
	plt.xlabel(r'Volumetric strain energy (J/mm$^3$)')
	plt.ylabel('Normalized probability')
	#plt.ylim((0,1e-5))

plt.show()