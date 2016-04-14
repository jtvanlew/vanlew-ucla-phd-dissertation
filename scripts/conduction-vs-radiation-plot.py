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

E = 120e9
nu = .24
R = 0.0005
ks = 2.4
Rstar = R/2
Estar = E/(2*(1-nu**2))

Hc = 2*ks*(3/4 * Rstar/Estar)**(1/3) * (10**(1/3))

ep = 0.5
sigma = 5.67e-8
A = 4*np.pi*R**2
Fij = 0.075
Hr = ep*sigma*A*Fij

T1 = np.linspace(300, 800, 100)
T1 = T1 + 273

for i, T in enumerate(T1):
    T2 = np.linspace(T - 100, T, 50)

    Qc = Hc * (T - T2)
    Qr = Hr * (T**4 - T2**4)

    Qr[0:-1] = Qr[0:-1]/Qc[0:-1]
    Qc[0:-1] = Qc[0:-1]/Qc[0:-1]
    
    for q in Qr:
        if q > 0.15:
            print(T-273)
            break
    # plt.figure(i)
    # plt.plot(T2[0:-1]-273, Qc[0:-1], linewidth = 2, color = color_idx[0], label='Conduction')
    # plt.plot(T2[0:-1]-273, Qr[0:-1], linewidth = 2, color = color_idx[1], label="Radiation")
    # plt.xlim(min(T2)-273, max(T2)-273)
    # plt.ylim(0, 1.1)
    # plt.xlabel(r"$T_2$ (C)")
    # plt.ylabel("Normalized heat transfer between pebbles")
    # plt.legend(loc='best')
    # plt.grid('on')
    # plt.savefig('../figures/conduction-vs-radiation/%s.png'%(max(np.round(T2-273,0))), format='PNG')

# plt.show()