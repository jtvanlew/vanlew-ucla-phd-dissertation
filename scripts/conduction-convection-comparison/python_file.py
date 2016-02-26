import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import spatial
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

df = pd.read_csv('dump_1400000.liggghts', delim_whitespace=True)
df_fc = pd.read_csv('dump.fc.1400000.liggghts', delim_whitespace=True)

df_fc.rename(columns={'c_fc[1]': 'x_1',
					  'c_fc[2]': 'y_1',
					  'c_fc[3]': 'z_1',
					  'c_fc[4]': 'x_2',
					  'c_fc[5]': 'y_2',
					  'c_fc[6]': 'z_2',
					  'c_fc[7]': 'id1',
					  'c_fc[8]': 'id2',
					  'c_fc[9]': 'periodic_flag',
					  'c_fc[10]': 'fn_x',
					  'c_fc[11]': 'fn_y',
					  'c_fc[12]': 'fn_z',
					  'c_fc[13]': 'contact_area',
					  'c_fc[14]': 'hflux',
					  }, 
			 inplace=True)
df_fc['fn'] = np.sqrt(df_fc['fn_x']**2 + df_fc['fn_y']**2 + df_fc['fn_z']**2)
df_fc['dx'] = df_fc['x_2'] - df_fc['x_1']
df_fc['dz'] = df_fc['z_2'] - df_fc['z_1']
df_fc['dy'] = pd.Series(np.zeros(len(df_fc['y_1'])), index=df_fc.index)

# first do a search for non-periodic pebbles and set the dy of those pebbles to y_2 - y_1
# then search for pebbles that are periodic but where the y_1 is in the positive half of the plane, so 'shift' y_2's location right by the y_lim length. 
# then search for pebbles that are periodic but where the y_1 is in the negative half of the plane, so 'shift' y_2's location left by the y_lim length. 
df_fc.loc[df_fc['periodic_flag'] == 0, 'dy'] = df_fc.loc[df_fc['periodic_flag'] == 0, 'y_2'] - df_fc.loc[df_fc['periodic_flag'] == 0, 'y_1']
df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] > 0), 'dy'] = df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] > 0), 'y_2'] + 0.004 - df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] > 0), 'y_1']
df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] < 0), 'dy'] = df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] < 0), 'y_2'] -         df_fc.loc[(df_fc['periodic_flag'] == 1) & (df_fc['y_1'] < 0), 'y_1'] - 0.004

df_fc['ds'] = np.sqrt(df_fc['dx']**2 + df_fc['dy']**2 + df_fc['dz']**2)

df_fc['T1'] = pd.Series(np.zeros(len(df_fc['y_1'])), index=df_fc.index)
df_fc['T2'] = pd.Series(np.zeros(len(df_fc['y_1'])), index=df_fc.index)




# LOAD FLUID DATA
df_fluid_load = pd.read_csv('x-walls-filled.csv')
df_fluid_load.rename(columns={'Points:0': 'x', 'Points:1': 'y','Points:2': 'z'}, inplace=True)
df_fluid = df_fluid_load.loc[(df_fluid_load['z']>=0) & (df_fluid_load['z']<=0.051)]

x_points = np.sort(np.asarray(list(set(df_fluid['x'].values))))
y_points = np.sort(np.asarray(list(set(df_fluid['y'].values))))
z_points = np.sort(np.asarray(list(set(df_fluid['z'].values))))




df['Tf'] = pd.Series(np.zeros(len(df['x'])), index=df.index)
for i, x in enumerate(x_points[0:-1]):
	for j, y in enumerate(y_points[0:-1]):
		for k, z in enumerate(z_points[0:-1]):
			Tf = df_fluid.loc[((df_fluid['x']==x_points[i]) | ( df_fluid['x']==x_points[i+1])) &
							  ((df_fluid['y']==y_points[j]) | ( df_fluid['y']==y_points[j+1])) &
							  ((df_fluid['z']==z_points[k]) | ( df_fluid['z']==z_points[k+1]))]['T'].mean()
			df.loc[(df['x'] >= x_points[i]) & (df['x'] <= x_points[i+1]) &
				   (df['y'] >= y_points[j]) & (df['y'] <= y_points[j+1]) &
				   (df['z'] >= z_points[k]) & (df['z'] <= z_points[k+1]), 'Tf'] = Tf


h = 2*0.2/0.001
Ap = 4*np.pi*0.0005**2
Qconv = h*Ap*(df['f_Temp[0]']-df['Tf'])


plt.figure(figsize=[6,12])
sc = plt.scatter(df['x'], df['z'], c = Qconv, s=40, alpha = 0.7)
cb = plt.colorbar(sc)
cb.set_label(r'Heat transfer INTO fluid per pebble (W)')
plt.xlabel('x span (m)')
plt.ylabel('z span (m)')
plt.xlim([-0.01, 0.01])
plt.ylim([0, df['z'].max()+.0005])
plt.tight_layout()
plt.show()


# program_starts = time.time()
# # this loop took about 20 seconds
for i in np.arange(1, 6001):
	df_fc.loc[df_fc['id1'] == i, 'T1'] = df.loc[df['id'] == i]['f_Temp[0]'].values[0]
	df_fc.loc[df_fc['id2'] == i, 'T2'] = df.loc[df['id'] == i]['f_Temp[0]'].values[0]

	Q_cond1 = df_fc.loc[df_fc['id1']==2, 'hflux'].sum()
	Q_cond2 = df_fc.loc[df_fc['id2']==2, 'hflux'].sum()
Q_cond[i-1] = Q_cond1 + Q_cond2
# now = time.time()
# print(now- program_starts)
