import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

store = pd.HDFStore('store-crushed-stagnant.h5')
df = store['df']  # load it


# CREATE A SUBSET THAT IS JUST THE PEBBLE BED (NO ENTRY AND EXIT REGION)
# CREATE A BACKUP OF THE FULL DF
df_orig = df
x_bed_min = df.loc[df['density']==0]['x'].min()
x_bed_max = df.loc[df['density']==0]['x'].max()

x_clip_1_df = df.loc[df['x']>=x_bed_min]
df = x_clip_1_df.loc[x_clip_1_df['x']<=x_bed_max]
# END CREATE A SUBSET THAT IS JUST THE PEBBLE BED ~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~ Y PROFILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v = np.zeros((df['y'].max())+1)
T = np.zeros((df['y'].max())+1)
phi = np.zeros((df['y'].max())+1)

for i in np.arange(df['y'].max()+1):
	y_subset = df.loc[df['y'] == i]
	#xy_subset = y_subset.loc[y_subset['x'] == 375/2]
	v_x = (y_subset['velocity:0']**2).mean()
	v_y = (y_subset['velocity:1']**2).mean()
	v_z = (y_subset['velocity:2']**2).mean()
	print(i)
	v[i] = np.sqrt(v_x + v_y + v_z)
	T[i] = y_subset['temperature'].mean()
	peb_vol = len(y_subset.loc[y_subset['density'] < 0.01])
	bed_vol = len(y_subset)
	phi[i] = peb_vol/bed_vol

with open('y-profiles-crushed-stagnant.pkl', 'wb') as f:
    pickle.dump([v, T, phi], f)

with open('y-profiles-crushed-stagnant.pkl', 'rb') as f:
    [v, T, phi] = pickle.load(f)




