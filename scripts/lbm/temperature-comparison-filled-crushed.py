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


def load_cfd_dem_csv(filename):
    df = pd.read_csv(filename)

    df.rename(columns={'Points:0': 'x', 'Points:1': 'y','Points:2': 'z'}, inplace=True)
    df = df.loc[(df['z']>=0) & (df['z']<=0.00858)]

    df['U'] = np.sqrt(df['U:0']**2 + df['U:1']**2 + df['U:2']**2)

    x_list = df['x']
    x_set = np.sort(np.asarray(list(set(x_list))))

    phi = np.zeros(len(x_set))
    T = np.zeros(len(x_set))
    U = np.zeros(len(x_set))
    for i, x in enumerate(x_set):
        df_yz = df.loc[df['x']==x]
        phi[i] = 1-(df_yz['voidfraction_0'].mean())
        U[i] = df_yz['U'].mean()
        T[i] = df_yz['T'].mean()-573
    x_cfd = x_set/0.001
    return(x_cfd, phi, U, T)

   

# CFD-DEM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_cfd, phi, U, T = load_cfd_dem_csv('filled-cfd-dem.csv')
plt.plot(x_cfd, T, color=color_idx[0], linewidth=2, label='CFD-DEM, filled')

x_cfd, phi, U, T = load_cfd_dem_csv('crushed-cfd-dem.csv')
plt.plot(x_cfd, T, color=color_idx[0], linewidth=2, linestyle='--', label='CFD-DEM, crushed')
# CFD-DEM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


L = 3/1000.
phi_k = 0.61
q_nuc = 8.e6

x_lbm = np.linspace(-3, 3, 121)
# LBM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with open('y-profiles-filled.pkl', 'rb') as f:
    [v, T_filled, phi_y] = pickle.load(f)
    T_filled = T_filled*109000
deltaT = np.max(T_filled) - np.min(T_filled)
keff_filled = q_nuc*phi_k*L**2/(2*deltaT)
print(keff_filled)

plt.plot(x_lbm, T_filled, color=color_idx[1], linewidth=2, label='LBM, filled')

with open('y-profiles-crushed.pkl', 'rb') as f:
    [v, T_crushed, phi_y] = pickle.load(f)
    T_crushed = T_crushed*130000
deltaT = np.max(T_crushed) - np.min(T_crushed)
keff_crushedT  = q_nuc*phi_k*L**2/(2*deltaT)
print(keff_crushedT )

plt.plot(x_lbm, T_crushed, color=color_idx[1], linewidth=2, linestyle='--', label='LBM, crushed')
# LBM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




plt.xlim([-3,3])
plt.grid('on')
plt.legend(loc='best')
plt.xlabel(r'Dimensionless span ($x/d_p$)')
plt.ylabel(r'Wall offset temperature, ($T - T_w$), (K)')
pngFile = '../../figures/lbm/cfd-lbm-T-profiles.png'
plt.savefig(pngFile)
plt.show()