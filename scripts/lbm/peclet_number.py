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


epsilon = np.linspace(0.05, 0.4, 100)
U = 0.05/epsilon
dp = 0.001
nuf = 8.5e-4
alphaf = 1.3e-3

Re = U *dp/ nuf
Pr = nuf/alphaf
Pe = Re*Pr

kdiss = (0.0075 * Pe**2/(2+1.1*Pe**0.6/Pr**0.27))*0.34
kdiss /= kdiss[-1]
plt.plot(1-epsilon, kdiss, linewidth = 2, color = color_idx[0])
plt.grid('on')
plt.xlabel('Packing fraction')
plt.ylabel('Normalized dispersive conductivity')
plt.show()
# calculated from LBM setup
# Pr = 0.665625

# Re-2 = 1.17371
# Pe-2 = 0.78125