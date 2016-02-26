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

store = pd.HDFStore('store-crushed.h5')
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

with open('y-profiles-crushed.pkl', 'wb') as f:
    pickle.dump([v, T, phi], f)

# with open('y-profiles-crushed.pkl', 'rb') as f:
#     [v, T, phi] = pickle.load(f)

x = np.linspace(-3, 3, 241)

fig, ax1 = plt.subplots()

ax1.plot(x, v/0.05, color = color_idx[0], linewidth=2)
ax1.set_xlabel(r'Dimensionless span ($x/d_p$)')
ax1.set_ylabel(r'Dimensionless velocity, $U/U_0$')
ax1.set_xlim([-3,3])
ax1.set_ylim([0, 7])
ax1.grid('on')
for tl in ax1.get_yticklabels():
    tl.set_color(color_idx[0])
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
ax2.plot(x, phi, color = color_idx[1], linewidth=2)
ax2.set_ylabel('Packing fraction')
for tl in ax2.get_yticklabels():
    tl.set_color(color_idx[1])
pngFile = '../../figures/lbm/y-phi-v-profiles-crushed.png'
plt.savefig(pngFile)
#~~~~~~~~~~ END Y PROFILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#~~~~~~~~~~ Z PROFILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v = np.zeros((df['z'].max())+1)
T = np.zeros((df['z'].max())+1)
phi = np.zeros((df['z'].max())+1)

for i in np.arange(df['z'].max()+1):
	z_subset = df.loc[df['z'] == i]
	#xy_subset = y_subset.loc[y_subset['x'] == 375/2]
	v_x = (z_subset['velocity:0']**2).mean()
	v_y = (z_subset['velocity:1']**2).mean()
	v_z = (z_subset['velocity:2']**2).mean()
	print(i)
	v[i] = np.sqrt(v_x + v_y + v_z)
	T[i] = z_subset['temperature'].mean()
	peb_vol = len(z_subset.loc[z_subset['density'] < 0.01])
	bed_vol = len(z_subset)
	phi[i] = peb_vol/bed_vol

with open('z-profiles-crushed.pkl', 'wb') as f:
    pickle.dump([v, T, phi], f)

# with open('z-profiles-crushed.pkl', 'rb') as f:
#     [v, T, phi] = pickle.load(f)

x = np.linspace(-2, 2, 160)


fig, ax1 = plt.subplots()

ax1.plot(x, v/0.05, color = color_idx[0], linewidth=2)
ax1.set_xlabel(r'Dimensionless span ($y/d_p$)')
ax1.set_ylabel(r'Dimensionless velocity, $U/U_0$')
ax1.set_xlim([-3,3])
ax1.set_ylim([0, 7])
ax1.grid('on')
for tl in ax1.get_yticklabels():
    tl.set_color(color_idx[0])
ax2 = ax1.twinx()
ax2.plot(x, phi, color = color_idx[1], linewidth=2)
ax2.set_ylabel('Packing fraction')
ax2.set_ylim([0, 1])
for tl in ax2.get_yticklabels():
    tl.set_color(color_idx[1])
pngFile = '../../figures/lbm/z-phi-v-profiles-crushed.png'
plt.savefig(pngFile)
#~~~~~~~~~~ END Z PROFILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# PACKED BED SUBSET FOR FINDING AVERAGE BED VELOCITY ~~~~~~~~~~~~~~~~~~
# x_bed_min = df.loc[df['density']==0]['x'].min()
# x_bed_max = df.loc[df['density']==0]['x'].max()

# x_clip_1_df = df.loc[df['x']>=x_bed_min]
# x_clip_df = x_clip_1_df.loc[x_clip_1_df['x']<=x_bed_max]

#T = 0.05 / x_clip_df['velocity:0'].mean()
T = 0.05 / df['velocity:0'].mean()
print('Tortuosity value = %s\n'%(np.round(T,3)))
# END PACKED BED SUBSET FOR FINDING AVERAGE BED VELOCITY ~~~~~~~~~~~~~~





# PRESURE DROP AND COZENY-KARMAN COMPARISON ~~~~~~~~~~~~~~~~~~~~~~~~~~~
rho_lb_i = df_orig.loc[df_orig['x']==0]['density'].mean()
rho_lb_o = df_orig.loc[df_orig['x']==df_orig['x'].max()]['density'].mean()

cs_squared = 1./3
ulb = 0.02
dx = 40./240
dt = dx*ulb

# adding 1/2 factor for scaling between rho_0
dP = (rho_lb_i-rho_lb_o)*cs_squared * dx**2 / dt**2
print('LBM pressure drop = %s Pa \n'%(np.round(dP,2)))


d_p = 0.001
mu = 4.17e-6
phi = 0.61
vs = 0.05

L = (x_bed_max - x_bed_min)*dx*d_p

dP_ck = 180.*mu/d_p**2 * (phi**2)/(1-phi)**3 * vs

print('CK pressure drop = %s Pa \n'%(np.round(dP_ck,2)))
# END PRESURE DROP AND COZENY-KARMAN COMPARISON ~~~~~~~~~~~~~~~~~~~~~~~


plt.show()