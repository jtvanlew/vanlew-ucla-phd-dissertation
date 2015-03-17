import numpy as np
import matplotlib.pyplot as plt
import sys, os
import csv

if len(sys.argv)>2:
    filenames = sys.argv[1:]
else:
    filenames = [sys.argv[1]]

i = 0
Fmax = np.zeros(len(filenames))
total_travel = np.zeros(len(filenames))
for loadname in filenames:
    print 'loading: '+loadname
    with open(loadname, 'rb') as f:
        data = list(csv.reader(f))
    s = data[0]
    F = data[1]

    # Find crush force
    Fmax[i] = F[-1]
    total_travel[i] = s[-1]
    print Fmax[i]

    plt.plot(s,F,label=loadname)
    plt.xlabel('Standard travel (mm)')
    plt.ylabel('Standard force (N)')
    i += 1

plt.figure(2)
plt.hist(Fmax)
print np.average(Fmax)
print np.average(total_travel)
plt.show()
