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

x = np.linspace(-3, 3, 241)
L = 3/1000.
phi = 0.61
q_nuc = 8.e6






# CFD-DEM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df = pd.read_csv('filled-cfd-dem.csv')

df.rename(columns={'Points:0': 'x', 'Points:1': 'y','Points:2': 'z'}, inplace=True)

df['U'] = np.sqrt(df['U:0']**2 + df['U:1']**2 + df['U:2']**2)

x_list = df['x']
x_set = np.sort(np.asarray(list(set(x_list))))

phi = np.zeros(len(x_set))
U = np.zeros(len(x_set))
for i, x in enumerate(x_set):
    df_yz = df.loc[df['x']==x]
    phi[i] = 1-(df_yz['voidfraction_0'].mean())
    U[i] = df_yz['U'].mean()
x_cfd = x_set/0.001
U_int_cfd = np.trapz(U, dx = 0.006/18)
print(U_int_cfd)
plt.plot(x_cfd, U, color=color_idx[0], linewidth=2, label='CFD-DEM')
# CFD-DEM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# LBM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with open('y-profiles-filled.pkl', 'rb') as f:
    [v, T_filled, phi_y] = pickle.load(f)
x_lbm = np.linspace(-3, 3, 241)
plt.plot(x_lbm, v/2, color=color_idx[1], linewidth=2, label='LBM')
v_int_lbm = np.trapz(v/2, dx = .001/40.)
print(v_int_lbm)
# LBM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




plt.xlim([-3,3])
plt.grid('on')
plt.legend(loc='best')
plt.xlabel(r'Dimensionless span ($x/d_p$)')
plt.ylabel(r'Wall offset temperature, ($T - T_w$), (K)')
pngFile = '../../figures/lbm/cfd-lbm-U-profiles-filled.png'
plt.savefig(pngFile)
plt.show()