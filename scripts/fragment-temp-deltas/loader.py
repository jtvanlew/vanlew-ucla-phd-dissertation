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

filename0 = 'x-walls/64percent/rstar125/5percent/dump_1300000.liggghts'
filename1 = 'x-walls/64percent/rstar125/5percent/dump_7680000.liggghts'
filename2 = 'x-walls/64percent/rstar125/5percent/dump_7880000.liggghts'

df_i  = pd.read_csv(filename0, delim_whitespace=True).sort('id')
df_Ti = pd.read_csv(filename1, delim_whitespace=True).sort('id')
df_Tf = pd.read_csv(filename2, delim_whitespace=True).sort('id')

df_i_pebs  = df_i.loc[df_Ti['id']<=6000]
df_Ti_pebs = df_Ti.loc[df_Ti['id']<=6000]
df_Tf_pebs = df_Tf.loc[df_Tf['id']<=6000]

df_i_fragments  = df_i.loc[df_Ti['id']>6000]
df_Ti_fragments = df_Ti.loc[df_Ti['id']>6000]
df_Tf_fragments = df_Tf.loc[df_Tf['id']>6000]

# must calculate travel better or else get a scatter from ovito with id vs displacement to load here
travel = np.loadtxt('id-displacement-scatter',skiprows=1)
# pull out only fragment travel
travel = travel[np.where(travel[:,0]>6000)]
travel = travel[travel[:,0].argsort()]

# create bins for making contours of temperature
x_bin_edges = np.linspace(df_i_pebs['x'].min(), df_i_pebs['x'].max(), 31, endpoint=True)
z_bin_edges = np.linspace(df_i_pebs['z'].min(), df_i_pebs['z'].max(), 61, endpoint=True)

# add a new column for the average temperature of a region (region defined by the bin edges)
df_Tf_fragments['Tbar'] = pd.Series(np.zeros(len(df_Tf_fragments['x'])), index=df_Tf_fragments.index)
# also add a 2d array
Tbar_array = np.zeros([len(z_bin_edges)-1, len(x_bin_edges)-1])
for i in np.arange(0, len(x_bin_edges)-1):
	for j in np.arange(0, len(z_bin_edges)-1):
		Tbar =df_Tf_pebs.loc[(df_Tf_pebs['x']>=x_bin_edges[i]) & (df_Tf_pebs['x']<=x_bin_edges[i+1]) & (df_Tf_pebs['z']>=z_bin_edges[j]) & (df_Tf_pebs['z']<=z_bin_edges[j+1])]['f_Temp[0]'].mean()
		Tbar_array[j,i] = Tbar
		df_Tf_fragments.loc[(df_Ti_fragments['x'] >= x_bin_edges[i]) & (df_Ti_fragments['x'] <= x_bin_edges[i+1]) &
			   (df_Ti_fragments['z'] >= z_bin_edges[j]) & (df_Ti_fragments['z'] <= z_bin_edges[j+1]), 'Tbar'] = Tbar


fragment_temperature_deltas = (df_Tf_fragments['f_Temp[0]']-573)/(df_Tf_fragments['Tbar']-573)
peb_temperature_deltas = df_Tf_pebs['f_Temp[0]']-df_Ti_pebs['f_Temp[0]']

# create a meshgrid for contour
x_grid = np.linspace(-10, 10, len(x_bin_edges)-1)
z_grid = np.linspace(0, 51, len(z_bin_edges)-1)
X, Y = np.meshgrid(x_grid, z_grid)

cs = plt.contourf(X, Y, Tbar_array)
cbar = plt.colorbar(cs)

x_insert = df_Ti_fragments['x']
x_pebs = df_i_pebs['x']
# plt.figure(2)
# plt.scatter(travel[:,1], fragment_temperature_deltas)

plt.show()