import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import spatial
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

# filename0 = 'x-walls/62percent/rstar125/5percent/dump_1400000.liggghts'
# filename1 = 'x-walls/62percent/rstar125/5percent/dump_9530000.liggghts'
# filename2 = 'x-walls/62percent/rstar125/5percent/dump_9730000.liggghts'

# heated full
filename0 = 'x-walls/64percent/rstar125/5percent/dump_1400000.liggghts'
# heated crushed
filename1 = 'x-walls/64percent/rstar125/5percent/dump_7880000.liggghts'

df_i  = pd.read_csv(filename0, delim_whitespace=True).sort('id')
df_f = pd.read_csv(filename1, delim_whitespace=True).sort('id')

# create bins for making contours of temperature
x_bin_edges = np.linspace(df_i['x'].min(), df_i['x'].max(), 21, endpoint=True)
z_bin_edges = np.linspace(df_i['z'].min(), df_i['z'].max(), 51, endpoint=True)

# also add a 2d array
Tbar_array = np.zeros([len(z_bin_edges)-1, len(x_bin_edges)-1])
for i in np.arange(0, len(x_bin_edges)-1):
	for j in np.arange(0, len(z_bin_edges)-1):
		Tbar_i = df_i.loc[(df_i['x']>=x_bin_edges[i]) & (df_i['x']<=x_bin_edges[i+1]) & (df_i['z']>=z_bin_edges[j]) & (df_i['z']<=z_bin_edges[j+1])]['f_Temp[0]'].mean()
		Tbar_f = df_f.loc[(df_f['x']>=x_bin_edges[i]) & (df_f['x']<=x_bin_edges[i+1]) & (df_f['z']>=z_bin_edges[j]) & (df_f['z']<=z_bin_edges[j+1])]['f_Temp[0]'].mean()
		Tbar_array[j,i] = Tbar_f - Tbar_i


# create a meshgrid for contour
x_grid = np.linspace(-10, 10, len(x_bin_edges)-1)
z_grid = np.linspace(0, 51, len(z_bin_edges)-1)
X, Y = np.meshgrid(x_grid, z_grid)

cs = plt.contourf(X, Y, Tbar_array)
cbar = plt.colorbar(cs)

plt.show()