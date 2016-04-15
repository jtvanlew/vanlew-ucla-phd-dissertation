from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

E = 120e9
nu = .24
R = 0.0005
ks = 2.4
Rstar = R/2
Estar = E/(2*(1-nu**2))



ep = 0.5
sigma = 5.67e-8
A = 4*np.pi*R**2
Fij = 0.075
Hr = ep*sigma*A*Fij

T1 = np.linspace(300, 800, 4)
T1 = T1 + 273

for i, T in enumerate(T1):
    T2 = np.linspace(T - 100, T-0.001, 1000)
    Fn = np.linspace(1,100, 1000)

    X, Y = np.meshgrid(Fn, T2)

    Hc = 2*ks*(3/4 * Rstar/Estar)**(1/3) * (X**(1/3))

    Qr = (Hr * (T**4 - Y**4))/(Hc * (T - Y))

    # Qr[0:-1] = Qr[0:-1]/Qc[0:-1]
    # Qc[0:-1] = Qc[0:-1]/Qc[0:-1]

    fig = plt.figure(i)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, (Y-273), Qr, cmap=cmaps.viridis,
                           linewidth=0, antialiased=False, shade=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(0, 0.5)
    ax.set_xlabel('Contact Force (N)')
    ax.set_ylabel('$Tj$ (C)')
    ax.set_zlabel("Normalized radiative exchange (Qr/Qc)")
    ax.set_title('%s (C)'%(np.round(T-273,0)))
    #plt.plot(T2[0:-1]-273, Qc[0:-1], linewidth = 2, color = color_idx[0], label='Conduction')
    #plt.plot(T2[0:-1]-273, Qr[0:-1], linewidth = 2, color = color_idx[1], label="Radiation")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.xlim(min(T2)-273, max(T2)-273)
    # plt.ylim(0, 1.1)
    # plt.xlabel(r"$T_2$ (C)")
    # plt.ylabel("Normalized heat transfer between pebbles")
    # plt.legend(loc='best')
    # plt.grid('on')
    fig.tight_layout()
    plt.savefig('../figures/conduction-vs-radiation/%s.png'%(max(np.round(T2-273,0))), format='PNG')

plt.show()