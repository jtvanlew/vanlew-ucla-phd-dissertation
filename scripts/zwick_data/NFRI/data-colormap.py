import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
#import pylab as P
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from scipy import polyfit
if len(sys.argv)>2:
	filenames = sys.argv[1:]
else:
	filenames = [sys.argv[1]]


Estand = 220.e9
nustand = 0.27

nupeb = 0.24
Epebbulk = 90.e9


endpoint = 0
j = 0
dp = np.zeros(len(filenames))
Fmax = np.zeros(len(filenames))
Fpoly = np.zeros((len(filenames),3))
kpeb = np.zeros(len(filenames))
err_peb = np.zeros(len(filenames))
E_rec = [[] for i in range(len(filenames))]
for loadname in filenames:
	# Open the file again to read the pebble diameter
	# print 'Finding diameter for: '+loadname
	f = open(loadname)
	# print loadname
	content = f.readlines()
	dp1 = content[1]
	dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.
	j +=1
plt.figure(4)
n, bins, patches = plt.hist(dp*1000, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'k', 'alpha', 0.5)

plt.xlabel('Pebble diameter (mm)')
plt.ylabel('Count')




cm = plt.get_cmap('hot')
cNorm = colors.Normalize(vmin=min(dp)*1000, vmax=max(dp)*1000)

scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig, (ax1, ax2) = plt.subplots(1, 2)
N = 15
ax1 = plt.subplot2grid((N,N), (0,0), colspan=N-2, rowspan=N)
ax2 = plt.subplot2grid((N,N), (0,N-2), rowspan=N)
ax1.set_axis_bgcolor('#D3D3D3')

j = 0

k = np.linspace(0,1,1000)
slope = []
for loadname in filenames:
	print 'Loading : '+loadname

	filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
	s = filedata[:,0]
	F = filedata[:,1]
	# to account for the initial force, find the starting value.
	# the number was varied to make better fits to the overall curves
	Fn0 = np.mean(F[0])*.5
	
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
	

	# find a k for every pebble
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

	# find the slope of the upper end of the curve
	slogfit = s[np.where(s>10**-3)]
	Flogfit = F[np.where(s>10**-3)]
	
	slope_temp, intercept = np.polyfit(np.log(slogfit), np.log(Flogfit), 1)
	slope.append(slope_temp)
	normColorVal = (dp[j] - min(dp))/(max(dp)-min(dp))
	color = cm(normColorVal)
	ax1.loglog(s,F,color=color)
	#ax1.loglog(s,Fhertz, color='g')
	#print F[0]
	j+=1
ax1.set_xlabel('Standard travel (mm)')
ax1.set_ylabel('Standard force (N)')
ax1.set_xlim((10**-5, endpoint))
ax1.set_ylim((0,50))
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
                                    norm=cNorm,
                                    orientation='vertical')
plt.figure(10)
n, bins, patches = plt.hist(slope, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'c', 'alpha', 0.75)
# plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
plt.xlabel(r'n for $s^n$ in Hertzian contact')
plt.ylabel('Count')
plt.show()
