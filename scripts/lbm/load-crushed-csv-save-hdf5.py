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

# 375 nodes in X (call it Z in thesis)
# 241 nodes in Y (call it X in thesis)
# 160 nodes in Z (call it Y in thesis)

# LOAD CSV FROM PARAVIEW AND APPEND UNSTRUCTURED X, Y, Z~~~~~~~~~~~~~~
df = pd.read_csv('crushed.csv')
df['x'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['y'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['z'] = pd.Series(np.zeros(len(df['velocity:0'])), index=df.index)
df['vmag'] = np.sqrt(df['velocity:0']**2 + df['velocity:1']**2 + df['velocity:2']**2)
count = 1
for j in np.arange(1,161):
	print(j)
	M1 = 375*241*(j-1)
	M2 = 375*241*(j)
	df['z'][M1:M2] = j-1
	for i in np.arange(1, 242):
		N2 = 375*count
		N1 = N2 - 375
		df['y'][N1:N2] = i-1
		count += 1
for i in np.arange(0,375):
	df['x'].iloc[i::375] = i
store['df'] = df  # save it