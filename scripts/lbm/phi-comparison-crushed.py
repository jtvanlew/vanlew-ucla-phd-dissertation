import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]


# LOAD CSV FROM PARAVIEW AND APPEND UNSTRUCTURED X, Y, Z~~~~~~~~~~~~~~
df = pd.read_csv('crushed-cfd-dem.csv')

df.rename(columns={'Points:0': 'x', 'Points:1': 'y','Points:2': 'z'}, inplace=True)

df['U'] = np.sqrt(df['U:0']**2 + df['U:1']**2 + df['U:2']**2)

x_list = df['x']
x_set = np.sort(np.asarray(list(set(x_list))))

phi = np.zeros(len(x_set))
T = np.zeros(len(x_set))
U = np.zeros(len(x_set))
for i, x in enumerate(x_set):
    df_yz = df.loc[df['x']==x]
    phi[i] = (1-(df_yz['voidfraction_0'].mean()))*2.1
    U[i] = df_yz['U'].mean()*2.5
    T[i] = df_yz['T'].mean()-573

print(np.trapz(phi,x_set)/0.006)
print(np.trapz(U,x_set)/0.006)
x = x_set/0.001

fig, ax1 = plt.subplots()

ax1.plot(x, phi, color = color_idx[1], linewidth=2,  label='CFD-DEM')
ax1.set_xlabel(r'Dimensionless span ($x/d_p$)')
ax1.set_ylabel(r'Packing fraction')
ax1.set_xlim([-3,3])
ax1.grid('on')

with open('y-profiles-crushed.pkl', 'rb') as f:
    [v, T, phi] = pickle.load(f)
v = v
x = np.linspace(-3, 3, 121)
print(np.trapz(phi,x)/6)
print(np.trapz(v,x)/6)
#fig, ax1 = plt.subplots()

ax1.plot(x, phi, color = color_idx[1], linestyle = '--', linewidth=2, label='LBM')
ax1.set_ylim([0, 1])
ax1.legend(loc='best')


pngFile = '../../figures/lbm/y-phi-profiles-crushed-cfd-dem.png'
plt.savefig(pngFile)

plt.show()