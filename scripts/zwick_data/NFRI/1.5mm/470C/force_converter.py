import numpy as np
import matplotlib.pyplot as plt
import sys, os
#import pylab as P


filenames = [sys.argv[1]]
offset = float(sys.argv[2])

endpoint = 0
j = 0
dp = np.zeros(len(filenames))
Fmax = np.zeros(len(filenames))
Wstar = np.zeros(len(filenames))
Fpoly = np.zeros((len(filenames),3))
kpeb = np.zeros(len(filenames))
err_peb = np.zeros(len(filenames))
E_rec = [[] for i in range(len(filenames))]
for loadname in filenames:
	# Open the file again to read the pebble diameter
	f = open(loadname)
	content = f.readlines()
	dp1 = content[1]
	dp[j] = float(dp1[15:(dp1.find("mm")-2)])/1000.
	j +=1

j = 0

k = np.linspace(0.01,1,1000)

filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
s = filedata[:,0]
F = filedata[:,1]
s2=[]
for i in s:
    if i > offset:
        s2.append(i)
s = np.array(s2)
s = s - offset
F = F[-len(s):]
minF = min(F)
F = F - minF

sCorrection = (1/1000.)*F
s = s - sCorrection


# Search for the maximum force, discard the rest of the plot
# after this point
maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
F = F[0:maxIndex]	
s = s[0:maxIndex]


# Make a polynomial fit
sfit = np.linspace(0,s[-1],1001)
Fpoly = np.poly1d(np.polyfit(s,F,3, full=False))
Ffit = Fpoly(sfit)

# Find crush force
Fmax[j] = F[-1]

print 'max force calculated: '+str(Fmax[j])

# endpoint for the plot
if s[-1] > endpoint:
    endpoint = s[-1]
	
 # Integrate Force over displacement for strain energy
W = np.trapz(F,s)

	
	# Find strain energy per pebble volume
Wstar[j] = W/(4/3. * np.pi * (dp[j]/(2.))**3.)
	

Estand = 220.e9
nustand = 0.24
Fn0 = 0.1
nupeb = 0.27
err = 1e5
Epebbulk = 120.e9
for i in k:
    # Using Hertz theory, find the predicted force for 
    # given displacement
    Epeb = i*Epebbulk
    Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
    # in the experiment, there is overlap causing the pre-load
    # use that preload to back-out the initial overlap
    so = ((3*Fn0/Estar)**2*(1/dp[j]))**(1./3)
    # Hertz force (with corrected initial strain)
    Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+sfit/1000.)**3)+Fn0
    diff = Fhertz - Ffit
    erri = np.linalg.norm(diff)
    if erri < err:
        err = erri
        kpeb[j] = i
        err_peb[j] = erri
Epeb = kpeb[j]*Epebbulk
E_rec[j] = "E = "+ str(Epeb/10**9) + " GPa"
Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
Fhertz = (1./3)*Estar*np.sqrt(dp[j]*(so+s/1000.)**3)+Fn0
#plt.figure(j)
#lt.plot(s, F, color='g')
#plt.plot(s, Fhertz, color='k')
# Plot all forces over displacements
labelstring = 'Pebble '+str(j)
labelstringH = "Hertz theory (E = "+str(Epeb/10.**9)+"GPa )"
plt.figure(100)
plt.plot(s, F, color='g')
#plt.plot(s, F, color=plotColor, label=labelstring)
plt.plot(s, Fhertz, color='k')
# plt.plot(sfit,Ffit)
# plt.legend()
plt.legend(('Experimental',"Hertz theory"))
#plt.title('KIT pebbles (0.2 ~ 0.6 mm) at room temperature')
plt.xlabel('Standard travel (mm)')
plt.ylabel('Standard force (N)')
plt.xlim((0, endpoint))
plt.ylim((0,100))



plt.show()