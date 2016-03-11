# This creates many slices of BMP images in the images/ subdirectory
# You must FIRST create a csv file out of the liggghts dump file (I don't
# remember why I did it this way, but that's how it is). You can get that
# by running the liggghts2comsol.py file in the mylpp src/ directory

# when you're ready to run this, it can be called by
# > python liggghts2palabos.py liggghts2comsolTofe2014.csv 0.02 0.015 0.03 0.0005 300
# for example

import pandas as pd 
import numpy as np 
import Image, ImageDraw
import sys, os
import matplotlib.pyplot as plt

try:
    os.mkdir('images')
except:
    pass

if len(sys.argv) != 6:
	print "error, the syntax is 'python liggghts2palabos.py filename xlim(diameters) ylim(diameters) Rp Resolution (pixels per diameter)'"
	print "example:  python liggghts2palabos.py liggghts2comsolTofe2014.csv 25 15 0.0005 20"
        sys.exit()
else:
	filename = sys.argv[1]
	data = pd.read_csv(filename, names=['x','y','z','r'])
	Rp = float(sys.argv[4])
	dp = 2.*Rp
	xlim = float(sys.argv[2])*dp
	ylim = float(sys.argv[3])*dp
	zlim = max(data.z)
	res = float(sys.argv[5])

	print 'z domain size: ' + str((zlim+5*Rp))
	print 'z dimension is adding 2.5 pebble radii to exit and 2.5 to entrance'
	
	Nx = xlim/dp * res
	Ny = Nx * ylim / xlim
	Nz = Nx * (zlim+5.*Rp) / xlim

	Nx += 1
	Ny += 1
	Nz += 1

	dx = dp/res
	dy = dx
	dz = dx

	shiftx = xlim/2.
	shifty = ylim/2.

	Nx, Ny, Nz = int(Nx), int(Ny), int(Nz)
	print "total # of nodes: " +str(Nx*Ny*Nz)
	print "resolution: "+str(res)
	print "pixels in x direction: "+str(Nx)
	print "pixels in y direction: "+str(Ny)
	print "pixels in z direction: "+str(Nz)
	
	z0 = np.linspace(-2.5*Rp,zlim+2.5*Rp,Nz)
	k = 0
	ep = np.zeros([Nz])
	xgrid = np.linspace(0,xlim,Nx, endpoint=True)
	ygrid = np.linspace(0,ylim,Ny, endpoint=True)
	xmap = np.zeros([Nz, Nx*Ny])
	
	for z0 in z0:
		m = 0
		xmap2d = np.zeros([Nx,Ny])
		for z in data.z:
			if abs(z-z0) < data.r[m]:
				R = np.sqrt((data.r[m]**2 - (z-z0)**2))

				xmin, xmax = data.x[m]+shiftx - R, data.x[m]+shiftx + R
				ymin, ymax = data.y[m]+shifty - R, data.y[m]+shifty + R
				xgrid_crop = xgrid[(xgrid >= xmin) & (xgrid <= xmax)]
				ygrid_crop = ygrid[(ygrid >= ymin) & (ygrid <= ymax)]

				for j in ygrid_crop:
					for i in xgrid_crop:
						D = np.sqrt((data.x[m]+shiftx-i)**2 + (data.y[m]+shifty-j)**2)
						if D < R:
							xmap2d[int(np.round(i/dx)),int(np.round(j/dy))] = 1
			m+=1		
		xmap[k,:] = xmap2d.flatten()
		k+=1

xmap2 = xmap.flatten()


np.savetxt('N200-lattice.dat', xmap2.astype(int),fmt='%i')