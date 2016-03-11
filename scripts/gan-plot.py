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
T, x = load_digitized_csv('gan.csv')

dt = T[np.argmax(T)] - T[0]
dx = (x[np.argmax(T)] - x[0])/1000.
qn = 8.e6
k = qn*dx**2/(2*dt)

print(k)
s = 40
plt.scatter(x,T, color=color_idx[0], facecolors = 'none', s=s, linewidth=2)
plt.grid('on')
plt.xlim([0, 22])
plt.xlabel('X dimension (mm)')
plt.ylabel("Temperature (C)")
plt.show()