# https://jef.works/STalign/notebooks/xenium-heimage-alignment.html
# import dependencies ---------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import STalignXe1ometifHE
# from STalign import STalign

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)
path = "~Desktop/raw_data_must"

# 1. Read data -------------------------
# Single cell data 1
# read in data
fname = path + '/Xenium/outs/cells.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())

# 5. Read data -------------------------
# Single cell data 2
# read in data
fname = '/Xenium/outs/cells.csv.gz'
df2 = pd.read_csv(fname)

# get cell centroids
xJ = np.array(df2['x_centroid'])
yJ = np.array(df2['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2,c='#ff7f0e')

# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=30)
ax = fig.axes[0]
ax.invert_yaxis()





