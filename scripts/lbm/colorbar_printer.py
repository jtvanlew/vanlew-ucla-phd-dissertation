import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib as mpl

import numpy as np
import pandas as pd

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)


# Make a figure and axes with dimensions as desired.
fig, ax = plt.subplots(figsize=[1, 10])


# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
bounds = [0, 1]
cb1 = mpl.colorbar.ColorbarBase(ax,
                                norm=norm,
                                ticks=bounds,
                                orientation='vertical')
plt.tight_layout()
for item in [fig, ax]:
    item.patch.set_visible(False)
plt.savefig('colormap.png', format='png', dpi=900)
plt.show()