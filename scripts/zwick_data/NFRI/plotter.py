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
# plt.figure(4)
# n, bins, patches = plt.hist(dp*1000, histtype='stepfilled')
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
# plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
# plt.xlabel('Pebble diameter (mm)')
# plt.ylabel('Count')


j = 0


k = np.linspace(0.01,1,1000)

for loadname in filenames:
	print 'Calculating elasticity reduction factor for: '+loadname
	#print loadname
	# if dp[j] < 0.45/1000:
	# 	plotColor = 'g'
	# elif dp[j] >= 0.45/1000 and dp[j] < 0.55/1000:
	# 	plotColor = 'r'
	# else:
	# 	plotColor = 'k'
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
	
	# Integrate Force over displacement for strain energy
	# W[j] = np.trapz(F,s)


	Estand = 220.e9
	nustand = 0.27
	Fn0 = 0.1
	nupeb = 0.24
	err = 1e5
	Epebbulk = 120.e9
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
	#plt.figure(j)
	#lt.plot(s, F, color='g')
	#plt.plot(s, Fhertz, color='k')
	
	#plt.savefig('singles/epsFile'+str(j))
	# Plot all forces over displacements
	labelstring = 'Pebble '+str(j)
	labelstringH = "Hertz theory (E = "+str(Epeb/10.**9)+"GPa )"
	plt.figure(100)
	plt.plot(s, F, color='g')
	#plt.plot(s, F, color=plotColor, label=labelstring)
	plt.plot(s, Fhertz, color='k')
	# plt.plot(sfit,Ffit)
	# plt.legend()
	plt.legend(('Experimental',"Hertz theory"))
	#plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
	plt.xlabel('Standard travel (mm)')
	plt.ylabel('Standard force (N)')
	plt.xlim((0, endpoint))
	plt.ylim((0,12))
	#plt.legend(['Experimental','Hertzian'])


	
	#plt.figure(2)
	#plt.plot(err_peb)

	j+=1







from scipy.stats import norm

# CHOOSE A PARAMETER TO COMMENT OUT
# data = dp/2
# data = kpeb*Epebbulk/10.**9
# data = Fmax
data = W


# Fit a normal distribution to the data:
mu, std = norm.fit(data)
print mu, std
plt.figure(1000)

# Plot the histogram.
plt.hist(data, histtype='stepfilled', alpha=0.6, color='c')

# Plot the PDF.
#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 100)
#p = norm.pdf(x, mu, std)
#plt.plot(x, p, 'k', linewidth=2)

plt.ylabel('Count')

# label for youngs modulus
# plt.xlabel("Young's modulus (GPa)")

# label for strain energy
plt.xlabel('Strain energy (mJ)')



# title for diameter
#title = "Fit results: mu = %.2f mm,  std = %.2f mm" % (mu*1000, std*1000)
#plt.title(title)








plt.figure(101)
plt.scatter(dp*1000, kpeb, s=60, color='c')
plt.xlim(min(dp*1000), max(dp*1000))
plt.ylabel('Elasticity reduction factor')
plt.xlabel('Pebble diameter (mm)')
if len(filenames)>1:
	plt.figure(102)

	n, bins, patches = plt.hist(kpeb, histtype='stepfilled')
	plt.setp(patches, 'facecolor', 'k', 'alpha', 0.75)
	plt.xlabel('Elasticity reduction factor')
	plt.ylabel('Count')

plt.show()