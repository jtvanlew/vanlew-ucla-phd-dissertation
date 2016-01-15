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

phi = [0.61, 0.62, 0.64]
keff = [.636, .709, .887]
keff_rough = [.398, .435, .523]

for i, keff in enumerate(keff):
	print((keff-keff_rough[i])/keff)

phi_ = np.linspace(0.6, 0.65, 100)
upper = np.ones(100)*0.3
lower = np.ones(100)*0.2

s = 40

fig, ax = plt.subplots()
ax.set_xlim([0.6, 0.65])
ax.set_ylim([0, 1])
ax.set_xlabel('Packing fraction')
ax.set_ylabel(r"$k_{eff}$ (W/m-K)")

ax.scatter(phi, keff, color=color_idx[0], facecolors = 'none', s=s, label='Smooth approx.', linewidth=2)
ax.scatter(phi, keff_rough, color=color_idx[1], facecolors = 'none', s=s, label='Roughness model', linewidth=2)
ax.fill_between(phi_, upper, lower, facecolor = 'black', alpha = 0.2, label='Experimental range')
plt.legend(loc=3)
plt.grid('on')
plt.savefig('../figures/initial_packing_study/keff-comparisons.png', format='PNG')

plt.show()