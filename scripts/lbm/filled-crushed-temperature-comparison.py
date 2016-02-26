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

with open('y-profiles-filled.pkl', 'rb') as f:
    [v, T_filled, phi_y] = pickle.load(f)
deltaT = np.max(T_filled) - np.min(T_filled)
keff_filled = q_nuc*phi*L**2/(2*deltaT)
with open('y-profiles-crushed.pkl', 'rb') as f:
    [v, T_crushed, phi_z] = pickle.load(f)
deltaT = np.max(T_crushed) - np.min(T_crushed)
keff_crushed = q_nuc*phi*L**2/(2*deltaT)

print(keff_filled, keff_crushed)

plt.plot(x, T_filled, color=color_idx[0], linewidth=2, label='Filled')
plt.plot(x, T_crushed, color=color_idx[1], linewidth=2, label='Crushed')
plt.xlim([-3,3])
plt.grid('on')
plt.legend(loc='best')
plt.show()