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
total_travel = np.zeros(len(filenames))
Fpoly = np.zeros((len(filenames),3))
kpeb = np.zeros(len(filenames))
err_peb = np.zeros(len(filenames))
E_rec = [[] for i in range(len(filenames))]
for loadname in filenames:
	# Open the file again to read the pebble diameter
	f = open(loadname)
	content = f.readlines()
	dp1 = content[1]
	dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.
	j +=1

# plt.figure(4)
# n, bins, patches = plt.hist(dp*1000, histtype='stepfilled')
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
# plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
# plt.xlabel('Pebble diameter (mm)')
# plt.ylabel('Count')


j = 0


k = np.linspace(0.01,1,1000)

for loadname in filenames:
	#print loadname
	if dp[j] < 0.45/1000:
		plotColor = 'g'
	elif dp[j] >= 0.45/1000 and dp[j] < 0.55/1000:
		plotColor = 'r'
	else:
		plotColor = 'k'
	filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
	s = filedata[:,0]
	F = filedata[:,1]
	
	# Search for the maximum force, discard the rest of the plot
	# after this point
	maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
	F = F[0:maxIndex]	
	s = s[0:maxIndex]


	total_travel[j] = s[-1]
	# Make a polynomial fit
	sfit = np.linspace(0,s[-1],1001)
	Fpoly = np.poly1d(np.polyfit(s,F,3, full=False))
	Ffit = Fpoly(sfit)

	# Find crush force
	Fmax[j] = F[-1]


	# endpoint for the plot
	if s[-1] > endpoint:
		endpoint = s[-1]
	
	# Integrate Force over displacement for strain energy
	W = np.trapz(F,s)

	
	# Find strain energy per pebble volume
	Wstar[j] = W/(4/3. * np.pi * (dp[j]/(2.))**3.)
	

	Estand = 220.e9
	nustand = 0.27
	Fn0 = 0.1
	nupeb = 0.24
	err = 1e5
	Epebbulk = 90.e9
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
	if s[-1] > 0.05:
		print loadname + " pebble has very long strain, review this data"
	Epeb = kpeb[j]*Epebbulk
	E_rec[j] = "E = "+ str(Epeb/10**9) + " GPa"
	Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
	Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+s/1000.)**3)+Fn0
	
	# Plot all forces over displacements
	labelstring = 'Pebble '+str(j)
	labelstringH = "Hertz theory (E = "+str(Epeb/10.**9)+"GPa )"
	plt.figure(1)
	plt.plot(s, F, color='g')
	plt.plot(s, Fhertz, color='k')
	plt.legend(('Experimental',"Hertz theory"))
	plt.xlabel('Standard travel (mm)')
	plt.ylabel('Standard force (N)')
	plt.xlim((0, endpoint))
	plt.ylim((0,12))
	
	j+=1
plt.savefig('exp_v_hertz')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# CRUSH FORCE HISTOGRAM PLOT
#
#
plt.figure(100)
plt.hist(Fmax)
plt.xlabel('Crush force (N)')
plt.ylabel('Count')
print 'F_max-average = ' + str(np.average(Fmax)) + '\n'
plt.savefig('fmax')

print 'average total travel =' + str(np.average(total_travel)) + '\n'
print 'average E= '+str(np.mean(kpeb*Epebbulk/10**9))
print "Young's modulii = " + str(kpeb*Epebbulk/10**9) + '\n'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PEBBLE DIAMETER VS. K SCATTER
#
#
plt.figure(101)
plt.scatter(dp*1000, kpeb, s=60, color='g')
plt.xlim(min(dp*1000), max(dp*1000))
plt.ylabel('Elasticity reduction factor')
plt.xlabel('Pebble diameter (mm)')
plt.savefig('f-v-d')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PEBBLE DIAMETER VS. FMAX SCATTER
#
#
plt.figure(102)
plt.scatter(dp*1000, Fmax, s=60, color='g')
plt.xlim(min(dp*1000), max(dp*1000))
plt.ylabel('Crush force (N)')
plt.xlabel('Pebble diameter (mm)')
plt.savefig('k-v-d')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# K HISTOGRAM PLOT
#
#
plt.figure(103)

n, bins, patches = plt.hist(kpeb, histtype='stepfilled')
plt.xlim((0,1))
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
plt.xlabel('Elasticity reduction factor')
plt.ylabel('Count')
plt.savefig('k_hist')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.show()
