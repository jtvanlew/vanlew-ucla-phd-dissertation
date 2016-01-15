import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate

def create_mesh(filename):
	df = pd.read_csv(filename)

	df2 = df.loc[df['voidfraction']<1]

	x_coords = []
	z_coords = []
	phi = []
	dup_count = 0
	for index, row in df2.iterrows():
		if row['Points:1'] == 0.0025:
			x_coords.append(row['Points:0'])
			z_coords.append(row['Points:2'])
			df_x = df.loc[df['Points:0'] == row['Points:0']]
			df_xz = df_x.loc[df_x['Points:2'] == row['Points:2']]
			phi.append(1 - df_xz['voidfraction'].mean())

	x_coords = np.asarray(x_coords)
	z_coords = np.asarray(z_coords)
	# Set up a regular grid of interpolation points
	xi, yi = np.linspace(x_coords.min(), x_coords.max(), 300), np.linspace(z_coords.min(), z_coords.max(), 300)
	xi, yi = np.meshgrid(xi, yi)

	# Interpolate
	rbf = scipy.interpolate.Rbf(x_coords, z_coords, phi, function='linear')
	zi = rbf(xi, yi)
	return x_coords, z_coords, zi, phi






# 62% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prefixes = ["x-62-0", 
			"x-62-r23-1",
			"x-62-r23-3",
			"x-62-r23-5",
]
labels = ['Packed bed', '1%','3%','5%']
xticks_labels = ['-0.010', '-0.005', '0',  '0.005', '0.010']
fig, ax = plt.subplots(2,2,figsize=(7,12), dpi=200,)# sharey=True)
fig2, ax2 = plt.subplots(2,2,figsize=(7,12), dpi=200,)# sharey=True)
# ax[0].set_title(r'Packing fraction $\phi$')
phi_array = []
j = 0
k = 0
for i, filename in enumerate(prefixes):
	x, z, zi, phi = create_mesh(filename+'.csv')
	phi_array.append(phi)
	# Set up a regular grid of interpolation points
	xi, yi = np.linspace(x.min(), x.max(), 300), np.linspace(z.min(), z.max(), 300)
	xi, yi = np.meshgrid(xi, yi)
	phi_diff = np.asarray(phi_array[k])-np.asarray(phi_array[0])
	print(filename + " largest local percent increase in packing fraction = %s"%(phi_diff.max()/0.62))
	print(filename + " largest local percent decrease in packing fraction = %s"%(phi_diff.min()/0.62))	
	# Interpolate
	rbf = scipy.interpolate.Rbf(x, z, phi_diff, function='linear')
	zi_deltas = rbf(xi, yi)

	
	if i > 1:
		j = 1	
		i -= 2
	if (i-1)%2==0:
		ax[i][j].set_xlabel('Location in x (m)')
		ax2[i][j].set_xlabel('Location in x (m)')
	p = ax[j][i].imshow(zi, vmin=0.55, vmax=0.7, origin='lower',
	           extent=[x.min(), x.max(), z.min(), z.max()])
	p2 = ax2[j][i].imshow(zi_deltas, vmin=-0.07, vmax=0.07, origin='lower',
	           extent=[x.min(), x.max(), z.min(), z.max()])
	ax[j][i].set_ylabel('Location in z (m)')
	ax[j][i].set_xticklabels(xticks_labels, rotation=35)
	ax[j][i].set_title(labels[k])
	ax2[j][i].set_ylabel('Location in z (m)')
	ax2[j][i].set_xticklabels(xticks_labels, rotation=35)
	ax2[j][i].set_title(labels[k])
	k+=1
	
vmin,vmax = p.get_clim()
#-- Defining a normalised scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax3 = fig.add_axes([0.85, 0.1, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm)
fig.subplots_adjust(left=0.15,right=0.85)

fig.savefig(prefixes[1]+'.png')

vmin,vmax = p2.get_clim()
#-- Defining a normalised scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax4 = fig2.add_axes([0.85, 0.1, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = mpl.colorbar.ColorbarBase(ax4, norm=cNorm)
fig2.subplots_adjust(left=0.15,right=0.85)

fig2.savefig(prefixes[1]+'-deltas.png')

# /62% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


















# 64% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



prefixes = ["x-64-0", 
			"x-64-r23-1",
			"x-64-r23-3",
			"x-64-r23-5",
]
labels = ['Packed bed', '1%','3%','5%']
xticks_labels = ['-0.010', '-0.005', '0',  '0.005', '0.010']
fig, ax = plt.subplots(2,2,figsize=(7,12), dpi=200,)# sharey=True)
fig2, ax2 = plt.subplots(2,2,figsize=(7,12), dpi=200,)# sharey=True)
# ax[0].set_title(r'Packing fraction $\phi$')
phi_array = []
j = 0
k = 0
for i, filename in enumerate(prefixes):
	x, z, zi, phi = create_mesh(filename+'.csv')
	phi_array.append(phi)
	# Set up a regular grid of interpolation points
	xi, yi = np.linspace(x.min(), x.max(), 300), np.linspace(z.min(), z.max(), 300)
	xi, yi = np.meshgrid(xi, yi)
	phi_diff = np.asarray(phi_array[k])-np.asarray(phi_array[0])
	print(filename + " largest local percent increase in packing fraction = %s"%(phi_diff.max()/0.64))
	print(filename + " largest local percent decrease in packing fraction = %s"%(phi_diff.min()/0.64))	
	# Interpolate
	rbf = scipy.interpolate.Rbf(x, z, phi_diff, function='linear')
	zi_deltas = rbf(xi, yi)

	
	if i > 1:
		j = 1	
		i -= 2
	if (i-1)%2==0:
		ax[i][j].set_xlabel('Location in x (m)')
		ax2[i][j].set_xlabel('Location in x (m)')
	p = ax[j][i].imshow(zi, vmin=0.55, vmax=0.7, origin='lower',
	           extent=[x.min(), x.max(), z.min(), z.max()])
	p2 = ax2[j][i].imshow(zi_deltas, vmin=-0.07, vmax=0.07, origin='lower',
	           extent=[x.min(), x.max(), z.min(), z.max()])
	ax[j][i].set_ylabel('Location in z (m)')
	ax[j][i].set_xticklabels(xticks_labels, rotation=35)
	ax[j][i].set_title(labels[k])
	ax2[j][i].set_ylabel('Location in z (m)')
	ax2[j][i].set_xticklabels(xticks_labels, rotation=35)
	ax2[j][i].set_title(labels[k])
	k+=1
	
vmin,vmax = p.get_clim()
#-- Defining a normalised scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax3 = fig.add_axes([0.85, 0.1, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm)
fig.subplots_adjust(left=0.15,right=0.85)

fig.savefig(prefixes[1]+'.png')

vmin,vmax = p2.get_clim()
#-- Defining a normalised scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax4 = fig2.add_axes([0.85, 0.1, 0.03, 0.8])
#-- Plotting the colormap in the created axes
cb1 = mpl.colorbar.ColorbarBase(ax4, norm=cNorm)
fig2.subplots_adjust(left=0.15,right=0.85)

fig2.savefig(prefixes[1]+'-deltas.png')

plt.show()