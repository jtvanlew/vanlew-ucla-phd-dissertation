# GENERIC POST-PROCESSING COMMANDS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from scipy import polyfit

if len(sys.argv)>2:
	filenames = sys.argv[1:]
else:
	filenames = [sys.argv[1]]



def find_diameters(filenames):
	dp = np.zeros(len(filenames))

	for j, loadname in enumerate(filenames):
		f = open(loadname)
		content = f.readlines()
		dp1 = content[1]
		dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.
	return dp
def plot_histogram(x, xlabel):
	plt.figure(np.random.random_integers(100,300))
	n, bins, patches = plt.hist(x, histtype='stepfilled')
	plt.setp(patches, 'facecolor', 'c')
	plt.ylabel('Count')
	plt.xlabel(xlabel)

def plot_scatter_dp_kpeb(dp, kpeb):
	plt.figure(np.random.random_integers(20,99))
	plt.scatter(dp*1000, kpeb, s=60, color='k')
	plt.xlim(min(dp*1000), max(dp*1000))
	plt.ylabel('Elasticity reduction factor')
	plt.xlabel('Pebble diameter (mm)')

def create_dp_colormap(dp):
	cm = plt.get_cmap('hot')
	cNorm = colors.Normalize(vmin=min(dp)*1000, vmax=max(dp)*1000)

	scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
	fig, (ax1, ax2) = plt.subplots(1, 2)
	N = 15
	ax1 = plt.subplot2grid((N,N), (0,0), colspan=N-2, rowspan=N)
	ax2 = plt.subplot2grid((N,N), (0,N-2), rowspan=N)
	ax1.set_axis_bgcolor('#D3D3D3')
	return cm, cNorm, ax1, ax2

def find_strain_energy(filenames):
	W = np.zeros(len(filenames))
	for j, loadname in enumerate(filenames):
		f = open(loadname)
		content = f.readlines()
		w1 = content[2]
		W[j] = float(w1[16:(w1.find("mJ")-2)])

def calculate_k_plot(filenames, Epebbulk, nupeb, dp, log_flag, Hertz_plot):
	cm, cNorm, ax1, ax2 = create_dp_colormap(dp)
	Estand = 220.e9
	nustand = 0.27

	Fmax = np.zeros(len(filenames))
	kpeb = np.zeros(len(filenames))
	E_rec = [[] for i in range(len(filenames))]
	
	endpoint = 0
	k = np.linspace(0,1,1000)
	slope = []
	for j, loadname in enumerate(filenames):
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
		# Find crush force
		Fmax[j] = F[-1]

		# endpoint for the plot
		if s[-1] > endpoint:
			endpoint = s[-1]
		
		# find a k for every pebble
		# Make a polynomial fit to compare against hertzian
		sfit = np.linspace(0,s[-1],1001)
		Fpoly = np.poly1d(np.polyfit(s,F,3, full=False))
		Ffit = Fpoly(sfit)
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
		if log_flag:
			ax1.loglog(s,F,color=color)
			if Hertz_plot:
				ax1.loglog(s,Fhertz, color='k')
		else:
			ax1.plot(s,F,color=color)
			if Hertz_plot:
				ax1.plot(s,Fhertz, color='k')


	ax1.set_xlabel('Standard travel (mm)')
	ax1.set_ylabel('Standard force (N)')
	ax1.set_xlim((10**-5, endpoint))
	#ax1.set_ylim((0,50))
	cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm, norm=cNorm, orientation='vertical')

	return kpeb, Fmax, E_rec, slope
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



dp = find_diameters(filenames)
kpeb, Fmax, Erec, slope = calculate_k_plot(filenames, 124.e9, 0.24, dp, log_flag=False, Hertz_plot=True)

plot_histogram(dp*1000, 'Pebble diameter (mm)')
plot_histogram(kpeb,    'Softening coefficient, k')
plot_histogram(Fmax,  	'Crush force (N)')
plot_histogram(slope,  r'n for $s^n$ in Hertzian contact')
plot_scatter_dp_kpeb(dp, kpeb)
plt.show()