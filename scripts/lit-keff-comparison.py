import numpy as np
import matplotlib.pyplot as plt

def load_digitized_csv(filename):
    import csv
    x = []
    y = []
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            x.append(row[0])
            y.append(row[1])
    x = np.array([float(i) for i in x])
    y = np.array([float(i) for i in y])
    return x, y

color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

T_cfd = [600]
k_64 = [1.203]
k_62 = [1.1047]

T_dig, k_e_dig_hat = load_digitized_csv('hatano.csv')
strain_dig, k_e_dig_tan = load_digitized_csv('tanigawa.csv')
T_tanigawa = [600, 600, 600, 600]

T_mandal = [600]
k_eff_mandal = [0.65]

s = 40

plt.scatter(T_dig, k_e_dig_hat, color=color_idx[0], facecolors = 'none', s=s, label='Hatano et al.', linewidth=2)
plt.scatter(T_tanigawa, k_e_dig_tan, color=color_idx[1], facecolors = 'none', s=s, label='Tanigawa et al.', linewidth=2)
plt.scatter(T_cfd, k_62, color=color_idx[2], facecolors = 'none', s=s, label=r'CFD-DEM, $\phi = 62\%$', linewidth=2)
plt.scatter(T_cfd, k_64, color=color_idx[3], facecolors = 'none', s=s, label=r'CFD-DEM, $\phi = 64\%$', linewidth=2)
plt.scatter(T_mandal, k_eff_mandal, color=color_idx[4], facecolors = 'none', s=s, label='Mandal et al.', linewidth=2)

plt.ylim([0, 1.5])
plt.xlabel("Temperature (C)")
plt.ylabel(r"$k_{eff}$ (W/m-K)")
plt.legend(loc=3)
plt.grid('on')
plt.savefig('../figures/initial_packing_study/keff-he-comparisons.png', format='PNG')
plt.show()