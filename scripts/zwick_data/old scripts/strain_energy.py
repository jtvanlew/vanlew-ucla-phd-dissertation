# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:24:29 2015

@author: Jon
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD DATA NAMES INTO AN ARRAY
if len(sys.argv)>2:
	filenames = sys.argv[1:]
else:
	filenames = [sys.argv[1]]
w = np.zeros(len(filenames))

# LOAD EACH FILE AND PULL OUT THE STRAIN ENERGY
for j, loadname in enumerate(filenames):
	# Open the file again to read the pebble diameter
	f = open(loadname)
	content = f.readlines()
	w1 = content[2]
	w[j] = float(w1[16:(w1.find("mJ")-2)])/1000. #J

# FIND SOME VALUES FROM THE STRAIN ENERGY
w_ave = np.mean(w)
w_min = np.min(w)


# CREATE A WEIGHT TO 'NORMALIZE' THE HISTOGRAM.
# THE INTENT IS TO HAVE THE BARS SUM TO 1 SO IT'S A PROBABILITY
# NOT JUST A 'COUNT' HISTOGRAM
weights = np.ones_like(w)/len(w)
hist, bin_edges = np.histogram(w, len(w), density=True)


# WITH THE HISTOGRAM DATA, WE CAN EASILY GET A CUMULATIVE DENSITY
# FUNCTION
cdf = np.cumsum(hist*np.diff(bin_edges))
# CREATE AN X FOR PLOTTING
x_w = np.linspace(np.min(w),np.max(w),len(w))*10**3
# PLOT THE DATA FROM EXPERIMENTS
plt.plot(x_w, cdf,'o',color='c',alpha=0.7, label='Experimental data')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CREATE SOME WEIBULL DATA TO FIT TO OUR EXPERIMENTAL VALUES
m = 1.6 #FITTING PARAMETER FOR WEIBULL DISTRIBUTION. MODIFY TO FIT DATA
N = 10000
w_fit = np.random.weibull(m, N)*(w_ave-w_min)+w_min
x_fit = np.linspace(np.min(w_fit),np.max(w_fit),100)*10**3


# AGAIN MAKE A NORMALIZED WEIGHT AND HISTOGRAM DATA TO MAKE A CDF
weights = np.ones_like(w_fit)/len(w_fit)
hist, bin_edges = np.histogram(w_fit, 100, density=True)
cdf2 = np.cumsum(hist*np.diff(bin_edges))

# PLOT THE WEIBULL FIT ON THE SAME FIGURE
plt.plot(x_fit,cdf2,color='k',alpha=0.7, label=r'Weibull fit with $\sigma = %s$'%(m))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# INSERT LEGEND, CLEAN PLOT LIMITS, LABEL, AND SHOW PLOT
plt.ylim([0, 1])
plt.xlim([np.min(w)*10**3, np.max(w)*10**3])
plt.ylabel("Probability")
plt.xlabel(r'Strain energy ($10^{-3}$ J)')
plt.legend(loc='best')
plt.show()