import matplotlib.pyplot as plt
import numpy as np

color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

grid = [3*41*20, 3*41*30, 3*41*40]
keff_62 = [1.047, 1.023, 1.019]
keff_64 = [1.203, 1.178, 1.176]


s = 40

fig, ax = plt.subplots()
# ax.set_xlim([0.6, 0.65])
# ax.set_ylim([0, 1])
ax.set_xlabel('Mesh count')
ax.set_ylabel(r"$k_{eff}$ (W/m-K)")

ax.scatter(grid, keff_64, color=color_idx[0], facecolors = 'none', s=s, label=r'$\phi_i = 0.64$', linewidth=2)
ax.scatter(grid, keff_62, color=color_idx[1], facecolors = 'none', s=s, label=r'$\phi_i = 0.62$', linewidth=2)

plt.ylim([0.9, 1.3])
plt.legend(loc=3)
plt.grid('on')
plt.savefig('../figures/cfd-grid-study.png', format='PNG')

plt.show()