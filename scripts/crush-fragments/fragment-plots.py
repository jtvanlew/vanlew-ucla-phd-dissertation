import numpy as np 
import matplotlib.pyplot as plt 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# rstar1 
# make an array of radius ratios, then find how many pebbles, N, conserve mass
rstar_1 = np.linspace(0.2,1,10000)
N = 1/rstar_1**3



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# rstar2
data = np.loadtxt('pebble_packing_radii.txt')
rstar_2 = {}
# load the data from the text file row by row
for row in data:
    n = data[:,0]
    rstar = data[:,1]

# push the loaded data into the dictionary format with key:value
# so we can say n pebbles has x ratio
for i, n in enumerate(n):
    rstar_2[str(int(n))] = rstar[i]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# calculate volume ratio
Vstar = []
Pf = []
for j, n in enumerate(N):
    n = str(int(n))
    if n in rstar_2:
        Pf.append(rstar_1[j])
        Vstar.append((rstar_1[j]/rstar_2[n])**3)
    else:
        print 'radius ratio for '+ n + ' pebbles is not in the dictionary'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I want to get a polynomial fit to the volume curve, so i'll do some math here
V2fit = Vstar[:np.argmax(Vstar)]
Pf2fit = Pf[:np.argmax(Vstar)]

pfit = np.linspace(np.min(Pf2fit), np.max(Pf2fit), 100)
Vfit = np.poly1d(np.polyfit(Pf2fit,V2fit,3))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plot the shit
plt.figure(1)
plt.plot(Pf, Vstar, label='Minimum volume')
plt.plot(pfit, Vfit(pfit),'c',label='3rd order polynomial fit')
plt.xlabel(r'Radius Ratio ($R_c/R_p$)')
plt.ylabel(r'Dimensionless Volume')
plt.xlim([0, 1])
plt.legend(loc='best')

plt.figure(2)
plt.plot(rstar_1, N)
plt.xlabel(r'Radius Ratio ($R_c/R_p$)')
plt.ylabel('Pebble Fragments (rounded to whole number)')

plt.show()