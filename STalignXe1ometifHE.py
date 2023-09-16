# https://jef.works/STalign/notebooks/xenium-heimage-alignment.html
# import dependencies ---------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# import STalign from upper directory
# skip next 2 lines if STalign.py in same folder as notebook
import sys
sys.path.append("..")
import STalign

from STalign import STalign

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)
path = "/Users/estelladong/Desktop/raw_data_must/"

# 1. Read H&E -------------------------
# Target is H&E staining image
# image_file = path + 'Xenium/outs/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png'
image_file = path + 'Xenium/outs/Xe_png_screenshot.png'
V = plt.imread(image_file)

# plot
fig,ax = plt.subplots()
ax.imshow(V)

# 2. RGB N*M*3 from 0 to 1 ------------
print(V.shape)
print(V.min())
print(V.max())

# 3. STalign image norm for outlier intensities -------------
Inorm = STalign.normalize(V)

print(Inorm.min())
print(Inorm.max())

fig,ax = plt.subplots()
ax.imshow(Inorm)

# 4. We will transpose Inorm to be a 3xNxM matrix for downstream analyses. ----------
I = Inorm.transpose(2,0,1)
print(I.shape)

YI = np.array(range(I.shape[1]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
XI = np.array(range(I.shape[2]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
extentI = STalign.extent_from_x((YI,XI))

# 5. We can also see the data to be aligned with pandas ------
# Single cell data to be aligned
fname = path + 'Xenium/outs/cells.csv.gz'
df = pd.read_csv(fname)
df.head()

# get cell centroid coordinates --------------------
xM = np.array(df['x_centroid'])
yM = np.array(df['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xM,yM,s=1,alpha=0.2)

## More plotting --------------------------------
# plot
fig,ax = plt.subplots()
ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(xM,yM,s=1,alpha=0.1)

############## Random plotting ##############
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show(block=True)




