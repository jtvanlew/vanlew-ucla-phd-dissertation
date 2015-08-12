# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:00:17 2014

@author: jon
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
#import pylab as P
import matplotlib.cm as mplcm
import matplotlib.colors as colors

E = 126e9
nu = 0.2415

Estand = 220.e9
nustand = 0.27

gammaStand = (1-nustand**2)/Estand
gammaPeb = (1-nu**2)/E

Estar = 1/(gammaPeb + gammaStand)

dp = np.linspace(0.2/1000,2./1000,25) #m

s = np.linspace(0,.03/1000,100) #m



cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=min(dp), vmax=max(dp))
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig, (ax1, ax2) = plt.subplots(1, 2)
N = 10
ax1 = plt.subplot2grid((N,N), (0,0), colspan=N-1, rowspan=N)
ax2 = plt.subplot2grid((N,N), (0,N-1), rowspan=N)
#ax1.set_axis_bgcolor((78./255,78./255,78./255))



for d in dp:
    F = 1./3 * Estar * np.sqrt(d*s**3)
    
    normColorVal = (d - min(dp))/(max(dp)-min(dp))
    color = cm(normColorVal)
    ax1.plot(s*1000,F,color=color)
ax1.set_xlabel('Standard travel (mm)')
ax1.set_ylabel('Standard force (N)')
ax1.set_xlim((0, max(s*1000)))
ax1.set_ylim((0,100))
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
                                    norm=cNorm,
                                    orientation='vertical')

plt.show()