import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

store = pd.HDFStore('store-crushed.h5')
df = store['df']  # load it


# CREATE A SUBSET THAT IS JUST THE PEBBLE BED (NO ENTRY AND EXIT REGION)
# CREATE A BACKUP OF THE FULL DF
df_orig = df
x_bed_min = df.loc[df['density']==0]['x'].min()
x_bed_max = df.loc[df['density']==0]['x'].max()
df = df.loc[(df['x']>=x_bed_min) & (df['x']<=x_bed_max)]
# END CREATE A SUBSET THAT IS JUST THE PEBBLE BED ~~~~~~~~~~~~~~~~~~~~~

y_bin_edges = np.linspace(df['y'].min(), df['y'].max(), df['y'].max()/60+1, endpoint=True)
x_bin_edges = np.linspace(df['x'].min(), df['x'].max(), (df['x'].max()-df['x'].min())/60+1, endpoint=True)

# df['temperature']
T_f = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])
T_tilde_f = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])

v_f = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])
v_tilde_f = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])

k_dis = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])
advective_term = np.zeros([len(x_bin_edges)-1, len(y_bin_edges)-1])
y_loc = np.zeros(len(y_bin_edges)-1)
x_loc = np.zeros(len(x_bin_edges)-1)

rhof = 0.538 #kg/m3
cpf = 5193 #j/kg-K
# # df['T_f'] = pd.Series(np.zeros(len(df['x'])), index=df.index)
for i in np.arange(0, len(x_bin_edges)-1):
	x_loc[i] = (x_bin_edges[i] + x_bin_edges[i+1])/2
	for j in np.arange(0, len(y_bin_edges)-1):
		y_loc = (y_bin_edges[j] + y_bin_edges[j+1])/2
		# finding intrinsic values of just fluid, so ignore the solid here. (solid -> density = 0)
		df = df.loc[df['density']>=0.01]

		# pull out a subset of dataframe points that are within the boundaries of the x, y box
		df_subset_ij = df.loc[(df['x']>=x_bin_edges[i]) & (df['x']<=x_bin_edges[i+1]) & (df['y']>=y_bin_edges[j]) & (df['y']<=y_bin_edges[j+1])]
		
		# Find intrinsic average of deviation from intrinsic average for temperature
		T_f[i,j] = df.loc[(df['x']>=x_bin_edges[i]) & (df['x']<=x_bin_edges[i+1]) & (df['y']>=y_bin_edges[j]) & (df['y']<=y_bin_edges[j+1])]['temperature'].mean()
		T_tilde_f[i,j] = np.abs(np.abs(T_f[i,j]) - np.abs(df_subset_ij['temperature'])).mean()

		# Find intrinsic average of deviation from intrinsic average for velocity
		# we only want to find the y-direction velocity to find the transverse conductivity
		v_f[i,j] = df.loc[(df['x']>=x_bin_edges[i]) & (df['x']<=x_bin_edges[i+1]) & (df['y']>=y_bin_edges[j]) & (df['y']<=y_bin_edges[j+1])]['velocity:1'].mean()
		v_tilde_f[i,j] = np.abs(np.abs(v_f[i,j]) - np.abs(df_subset_ij['velocity:1'])).mean()

		# finding the intrinsic temperature gradient
		temperature_edge_left  = df.loc[(df['y']==df_subset_ij['y'].min())]['temperature'].mean()
		temperature_edge_right = df.loc[(df['y']==df_subset_ij['y'].max())]['temperature'].mean()
		nabla_y_temperature = np.abs(temperature_edge_left - temperature_edge_right)/(250e-6)

		# finally, find k_disp
		k_dis[i,j] = np.abs(rhof*cpf*0.39*T_tilde_f[i,j]*v_tilde_f[i,j]/nabla_y_temperature)

		# calculate the advective term in the transverse direction
		advective_term[i,j] = 0.39*rhof*cpf*np.abs(v_f[i,j])*nabla_y_temperature

x_grid = np.linspace(0, 7.1, len(x_bin_edges)-1)
y_grid = np.linspace(-3, 3, len(y_bin_edges)-1)
X, Y = np.meshgrid(y_grid, x_grid)

plt.figure(1)
cs = plt.contourf(X, Y, k_dis)
cbar = plt.colorbar(cs)
plt.xlabel(r'Dimensionless span ($x/d_p$)')
plt.ylabel(r'Dimensionless height ($z/d_p$)')
cbar.ax.set_ylabel(r'Dispersion conductivity, $k_{dis}$, (W/m-K)')

plt.figure(2)
cs = plt.contourf(X, Y, advective_term)
cbar = plt.colorbar(cs)
plt.xlabel(r'Dimensionless span ($x/d_p$)')
plt.ylabel(r'Dimensionless height ($z/d_p$)')
cbar.ax.set_ylabel(r'Advective energy, (W/m3)')

plt.figure(3)
cs = plt.contourf(X, Y, v_f)
cbar = plt.colorbar(cs)
plt.xlabel(r'Dimensionless span ($x/d_p$)')
plt.ylabel(r'Dimensionless height ($z/d_p$)')
cbar.ax.set_ylabel(r'Transverse velocity, (m/s)')
plt.show()