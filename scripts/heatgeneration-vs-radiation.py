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

R = 0.0005

V = 4/3 * np.pi * R**3
Qh = 8e6
Qn = V*Qh

ep = 0.7
sigma = 5.67e-8
A = 4*np.pi*R**2
Fij = 0.075
Hr = ep*sigma*A*Fij


x = np.linspace(0, 1, 11)
Tp = (1-x**2)*400
dT = np.diff(Tp)

T1 = np.linspace(850, 450, 10)
Qr = np.zeros(len(T1))
for i, T in enumerate(T1):
    Qr[i] = Hr * ((T+273)**4 - (T+dT[i]+273)**4)

Q = Qr/Qn





# fig, ax = plt.subplots()
# ax.plot(T1, Q, linewidth=2, color=color_idx[0], label="10 peb")

# x = np.linspace(0, 1, 51)
# Tp = (1-x**2)*400
# dT = np.diff(Tp)

# T1 = np.linspace(850, 450, 50)
# Qr = np.zeros(len(T1))
# for i, T in enumerate(T1):
#     Qr[i] = Hr * ((T+273)**4 - (T+dT[i]+273)**4)

# Q = Qr/Qn

# ax.plot(T1, Q, linewidth=2, color=color_idx[1], label="50 peb")

# ax.set_xlabel("Bed Temperature (C)")
# ax.set_ylabel("Q* = Qr/Qn")
# ax.grid('on')

# fig2, ax2 = plt.subplots()
# ax2.plot(x, Tp+450, linewidth=2, color=color_idx[1])
# ax2.set_xlabel("x")
# ax2.set_ylabel("T")
# ax2.grid('on')

# plt.legend(loc='best')




fig, ax1 = plt.subplots()
ax1.plot(np.linspace(0,1,10), Q, linewidth=2, color=color_idx[0], label="10 peb")
x = np.linspace(0, 1, 51)
Tp = (1-x**2)*400
dT = np.diff(Tp)
T1 = np.linspace(850, 450, 50)
Qr = np.zeros(len(T1))
for i, T in enumerate(T1):
    Qr[i] = Hr * ((T+273)**4 - (T+dT[i]+273)**4)
Q = Qr/Qn
ax1.plot(np.linspace(0,1,50), Q, linewidth=2, linestyle='--', color=color_idx[0], label="50 peb")
ax1.set_xlabel("Non-dimensional pebble bed span")
ax1.set_ylabel("Q* = Qr/Qn")
for tl in ax1.get_yticklabels():
    tl.set_color(color_idx[0])
ax2 = ax1.twinx()
ax2.plot(x, Tp+450, linewidth=2, color=color_idx[1])
ax2.set_ylabel('T (C)')
for tl in ax2.get_yticklabels():
    tl.set_color(color_idx[1])
ax1.grid('on')
plt.legend(loc='best')
plt.show()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, (Y-273), Qr, cmap=cmaps.viridis,
    #                        linewidth=0, antialiased=False, shade=True)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_zlim(0, 0.5)
    # ax.set_xlabel('Contact Force (N)')
    # ax.set_ylabel('$Tj$ (C)')
    # ax.set_zlabel("Normalized radiative exchange (Qr/Qc)")
    # ax.set_title('%s (C)'%(np.round(T-273,0)))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
#    ax.plot(T2-273, Qr, linewidth = 2, color = color_idx[i], label="T1 = %s C"%(np.round(T,0)-273))
# plt.xlim(min(T2)-273, max(T2)-273)
# plt.ylim(0, 1.1)
# plt.xlabel(r"$T_2$ (C)")
# plt.ylabel("Ratio of radiation to conductance heat transfer")
# plt.legend(loc='best')
# plt.grid('on')
# fig.tight_layout()
# # plt.savefig('../figures/conduction-vs-radiation/%s.png'%(max(np.round(T2-273,0))), format='PNG')

# plt.show()