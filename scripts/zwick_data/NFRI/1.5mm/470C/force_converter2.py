import numpy as np
import matplotlib.pyplot as plt
import sys

# EXAMPLE RUN OF CODE:
# > python force_converter.py <pebbleDataFile> <offset>
#
# example:
# > python force_converter.py singlePebble_1.TXT 0.078

filenames = [sys.argv[1]]

endpoint = 0

for loadname in filenames:
	f = open(loadname)
	content = f.readlines()

filedata = np.loadtxt(loadname,skiprows=6,delimiter=',')
# FILE LOADED WITH RAW DATA -- NEED TO REMOVE BELLOWS FORCE & PISTON STRAIN
s = filedata[:,0]
F = filedata[:,1]


plt.figure(1)
plt.plot(s,F)
plt.title('raw data')
# FIRST DISCARD ALL DATA AFTER BREAK
# Search for the maximum force, discard the rest of data
maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
F = F[0:maxIndex]	
s = s[0:maxIndex]

plt.figure(2)
plt.plot(s,F)
plt.title('discarded data after break')
# FIND THE FORCE AT EACH POINT OF DISPLACEMENT, s, AS A FUNCTION OF THE
# LINEAR STIFFNESS. SUBTRACT THIS FROM OVERALL FORCE
bellows_m = 1.91 #N/mm
Fbellows = bellows_m * s
F = F - Fbellows

plt.figure(3)
plt.plot(s,F)
plt.title('corrected for bellows')

df = np.diff(F)


# LOWER THIS VALUE IF TOO MUCH OFFSET IS BEING DELETED FROM DATA
#
err = 0.002
#
#



j = 0
for dff in df:
    if dff > err:
        offset = s[j]
        offset_force = F[j]
        break
    j += 1
print 'amount of removed offset: ' + str(offset)
print 'force value at offset: ' + str(offset_force)

# THE DATA RECORDED TRAVEL OF THE COMPRESSING BELLOWS. THIS ISN'T IMPORTANT 
# SO DELETE ALL DATA POINTS THAT ARE LESS THAN THE INPUT OFFSET. THIS OFFSET 
# VALUE IS FOUND FROM THE ERROR VALUE ABOVE
s2=[]
for i in s:
    if i > offset:
        s2.append(i)
s = np.array(s2)
s = s - offset
F = F[-len(s):]

plt.figure(4)
plt.plot(s,F)
plt.title('offset travel of bellows removed')

# USE DATA FROM CORRECTION CURVE TO REMOVE STRAIN FROM PISTON
piston_m = 455. # @ 470 C
# ZERO OUT THE FORCE
F = F - min(F)
piston_x = F/piston_m


s = s-piston_x

plt.figure(5)
plt.plot(s,F)
plt.title('corrected for travel of piston')

# Find crush force
Fmax = F[-1]

print 'max force: '+str(Fmax)

# endpoint for the plot
if s[-1] > endpoint:
    endpoint = max(s)
	



# Make a polynomial fit
sfit = np.linspace(0,s[-1],1001)
Fpoly = np.poly1d(np.polyfit(s,F,4, full=False))
Ffit = Fpoly(sfit)



k = np.linspace(0.,1,1000)
dp= 1.5
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
    so = ((3*Fn0/Estar)**2*(1/dp))**(1./3)
    # Hertz force (with corrected initial strain)
    Fhertz = (1./3)*Estar*np.sqrt(dp*(so+sfit/1000.)**3)+Fn0
    diff = Fhertz - Ffit
    erri = np.linalg.norm(diff)
    if erri < err:
        err = erri
        kpeb = i
        err_peb = erri
    elif i == 1:
        print 'no fit, setting k_peb = 1'
        kpeb = 1

Epeb = kpeb*Epebbulk
E_rec = "E = "+ str(Epeb/10**9) + " GPa"
Estar = 1./((1.-nupeb**2)/Epeb + (1.-nustand**2)/Estand)
Fhertz = (1./3)*Estar*np.sqrt(dp*(so+s/1000.)**3)+Fn0





plt.figure(100)
labelstring = 'Pebble '+str(j)
labelstringH = "Hertz theory (E = "+str(Epeb/10.**9)+"GPa )"

plt.plot(s, F, color='g')
plt.plot(s, Fhertz, color='k')

plt.legend(('Experimental',"Hertz theory"))
plt.xlabel('Standard travel (mm)')
plt.ylabel('Standard force (N)')
plt.xlim((0, endpoint))
plt.ylim((0,100))

plt.show()