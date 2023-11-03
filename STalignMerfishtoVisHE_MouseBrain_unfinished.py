## import dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import torch
import plotly
import requests

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)

# # OPTION A: import STalign after pip or pipenv install
from STalign import STalign

path = "/Users/estelladong/Desktop/raw_data_must/"

# Single cell data 1
# read in data
## Download source: https://console.cloud.google.com/storage/browser/public-datasets-vizgen-merfish/datasets/mouse_brain_map/
## BrainReceptorShowcase/Slice2/Replicate3;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))
## &prefix=&forceOnObjectsSortingFiltering=false
fname = path + 'Merfish_MouseBrain/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv'
df1 = pd.read_csv(fname)
print(df1.head())

# get cell centroid coordinates
xI = np.array(df1['center_x'])
yI = np.array(df1['center_y'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)
#ax.set_aspect('equal', 'box')
plt.show()

## --------------------------------------------------------------
# Note that points are plotting with the origin at bottom left while images are typically plotted with
# origin at top left so we’ve used invert_yaxis() to invert the yaxis for visualization consistency.
# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI, yI, dx=30)

ax = fig.axes[0]
ax.invert_yaxis()
plt.show()


# make it colorful
print("The initial shape of I is {}".format(I.shape))
I = np.vstack((I, I, I)) # make into 3xNxM
print("The range of I is {} to {}".format(I.min(), I.max() ))

# normalize
I = STalign.normalize(I)
print("The range of I after normalization is {} to {}".format(I.min(), I.max() ))

# double check size of things
print("The new shape of I is {}".format(I.shape))


## Read in Visium -------------------------------------------------
image_file = path + "Visium_MouseBrain/spatial/tissue_hires_image.png"
V = plt.imread(image_file)

# plot
fig,ax = plt.subplots()
ax.imshow(V)
plt.show()


print("The initial shape of V is {}".format(V.shape))
print("The range of V is {} to {}".format(V.min(), V.max() ))

Vnorm = STalign.normalize(V)
print("The range of V after normalization is {} to {}".format(Vnorm.min(), Vnorm.max() ))


## ----------------------------------------------------------------
J = Vnorm.transpose(2,0,1)
print("The new shape of J is {}".format(J.shape))

YJ = np.array(range(J.shape[1]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
XJ = np.array(range(J.shape[2]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.

# plot
# get extent of images
extentJ = STalign.extent_from_x((YJ,XJ))
extentI = STalign.extent_from_x((YI,XI))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI)
ax[0].invert_yaxis()
ax[0].set_title('source', fontsize=15)
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)
ax[1].set_title('target', fontsize=15)
plt.show()








