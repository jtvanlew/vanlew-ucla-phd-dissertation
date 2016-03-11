# This file wants scatter data from a .txt file that has been created by Ovito
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#loadfile1 = input('load file 1 name...')
loadfiles = ['z-scatter-0.20',
			 'z-scatter-0.25',
			 'z-scatter-0.35',
			 'z-scatter-0.50',
]
labels = ['r* = 0.20','r* = 0.25','r* = 0.35','r* = 0.50']

#fig, ax = plt.subplots(1)
fig, ax = plt.subplots()
alpha = 1
ax.set_ylabel(r'Packing fraction at height z')
ax.set_xlabel(r'Nondimensional height location, $z/d_p$')

#ax.set_axis_bgcolor('#595959')
# ax.set_xlim((0.0001, 1))
# ax.set_ylim((0.001, 1.2))
n = np.size(loadfiles)
color = plt.cm.jet(np.linspace(0,1,n))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create an inset axis 
axins = zoomed_inset_axes(ax,    # name of parent axes
						  4,   # zoom level
						  loc=5, # location of inset placement
						  )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]
             ]

for j, loadfile in enumerate(loadfiles):
	data = np.loadtxt(loadfile)

	for row in data:
		r = data[:,0]
		x = data[:,1]
	
	Nbins = 20
	counter = np.zeros([Nbins])
	vbin = np.zeros([Nbins])
	xbin = np.zeros([Nbins])
	xspan = np.abs(np.max(x)) + np.abs(np.min(x))
	dx = xspan/Nbins

	x0 = 0
	for i in np.arange(1,Nbins):
		x1 = x0 + dx
		xbin[i] = x0 + dx/2.
		vbin[i] = 0
		count = 0
		index = 0
		for xvalue in x:
			if xvalue >= x0 and xvalue < x1:
				vbin[i] += (4./3 * np.pi * r[index]**3)
				count += 1
			index += 1
		vbin[i] /=(dx*.000375)
		counter[i] = count
		x0 = x1

	
	ax.plot(xbin/.001, vbin,
				color=color_idx[j], alpha=alpha,
				#linewidth = 0,
				label=labels[j],)
	axins.plot(xbin/.001, vbin,
				color=color_idx[j], alpha=alpha,
				#linewidth = 0,
				label=labels[j],)
	# plt.figure(2)
	# plt.hist(displa_small,color=color[j])


ax.legend(loc='best')
# sub region of the original image
x1, x2, y1, y2 = 0, 2, 0.55, 0.65
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# plt.xticks(visible=False)
# plt.yticks(visible=False)
# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")







plt.show()
