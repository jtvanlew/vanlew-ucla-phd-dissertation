import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
color_idx = [[0./255,107./255,164./255], 
             [255./255, 128./255, 14./255], 
             [171./255, 171./255, 171./255], 
             [89./255, 89./255, 89./255],
             [44./255, 160./255, 44./255],
             [95./255, 158./255, 209./255],
             [200./255, 82./255, 0./255],
             [255./255, 152./255, 150./255]]

store = pd.HDFStore('store-crushed.h5')
# df = store['df']  # load it

# 287 nodes in X (call it Z in thesis)
# 121 nodes in Y (call it X in thesis)
#  81 nodes in Z (call it Y in thesis)

# LOAD CSV FROM PARAVIEW AND APPEND UNSTRUCTURED X, Y, Z~~~~~~~~~~~~~~
df = pd.read_csv('crushed-lbm.csv')
df['x'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['y'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['z'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['vmag'] = np.sqrt(df['velocity:0']**2 + df['velocity:1']**2 + df['velocity:2']**2)
count = 1
for j in np.arange(1,82):
	print(j)
	M1 = 287*121*(j-1)
	M2 = 287*121*(j)
	df['z'][M1:M2] = j-1
	for i in np.arange(1, 122):
		N2 = 287*count
		N1 = N2 - 287
		df['y'][N1:N2] = i-1
		count += 1
for i in np.arange(0,287):
	df['x'].iloc[i::287] = i
store['df'] = df  # save it