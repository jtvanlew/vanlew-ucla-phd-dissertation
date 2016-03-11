import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)


store = pd.HDFStore('store-filled.h5')
df = store['df']  # load it

x_grid = np.linspace(-3, 3, 121)
y_grid = np.linspace(-2, 2, 81)

X, Y = np.meshgrid(x_grid, y_grid)

z_slices = np.linspace(0, 283, 284)
#z_slices = np.linspace(0, 374, 375)

v = np.sqrt(df['velocity:0']**2 + df['velocity:1']**2 + df['velocity:2']**2)/0.05
levels = np.linspace(0, 35, 200, endpoint=True)
ticks = np.linspace(0, 35, 15, endpoint = True)

for k, z in enumerate(z_slices):
	print("Running slice %s of %s"%(k,len(z_slices)-1))
	df_subset = df.loc[df['x'] == np.round(z,0)]
	Z = np.zeros([len(y_grid), len(x_grid)])
	for i, x in enumerate(x_grid):
		df_subset_x = df_subset.loc[df_subset['y'] == i]
		for j, y in enumerate(y_grid):
			Z[j, i] = df_subset_x.loc[df_subset_x['z'] == j]['vmag']/0.05
	
	plt.figure(k)
	cs = plt.contourf(X, Y, Z, levels=levels)
	cbar = plt.colorbar(cs, ticks=ticks)
	plt.xlabel(r'Dimensionless span ($x/d_p$)')
	plt.ylabel(r'Dimensionless width ($y/d_p$)')
	cbar.ax.set_ylabel(r'Dimensionless velocity ($U/U_0$)')
	pngFile = 'images-filled/contour-%s.png'%(k)
	plt.savefig(pngFile)
plt.show()