import numpy as np
import matplotlib.pyplot as plt
import sys, os
import csv

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


# Search for the maximum force, discard the rest of the plot
# after this point
maxIndex = max(enumerate(F),key=lambda x: x[1])[0]
print F[maxIndex]	
s = s[0:maxIndex]

print F[-1]
data = [s, F]
saveFile = sys.argv[1] + '_edit.csv'
writer = csv.writer(open(saveFile, 'w'))
for row in data:
    writer.writerow(row)