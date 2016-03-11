import numpy as np
import matplotlib.pyplot as plt

color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

h = 680
dp = 0.001

k = np.linspace(0.01, 1, 100)
kirr = k*2.4

Bi = h*dp/kirr

Bi = Bi/Bi[-1]

hp = h/(1+Bi/5.)
hp = hp/hp[-1]

fig, ax1 = plt.subplots()
ax1.plot(k, Bi, linewidth=2, color=color_idx[0])
ax1.set_xlabel(r'$k_{irr}/k_{unirr}$')
ax1.set_ylabel('Normalized Biot Number (Bi/Bi,unirr)')
for tl in ax1.get_yticklabels():
    tl.set_color(color_idx[0])
ax2 = ax1.twinx()
ax2.plot(k, hp, linewidth=2, color=color_idx[1])
ax2.set_ylabel(r'Normalized Modified heat transfer coefficient (h/h,unirr)')
for tl in ax2.get_yticklabels():
    tl.set_color(color_idx[1])
ax1.grid('on')
pngFile = 'biot-changes.png'
plt.savefig(pngFile)
plt.show()