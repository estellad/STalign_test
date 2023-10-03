# https://jef.works/STalign/notebooks/xenium-heimage-alignment.html
# import dependencies ---------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scanpy

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
image_file = path + 'Xenium/outs/Xe_Fiji_downsize.png'
# image_file = path + 'Visium/outs/spatial/tissue_image_Fiji_scaled.tif'
V = plt.imread(image_file)

# plot
fig,ax = plt.subplots()
ax.imshow(V)
plt.show()

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
plt.show()

# 4. We will transpose Inorm to be a 3xNxM matrix for downstream analyses. ----------
I = Inorm.transpose(2,0,1)
print(I.shape)

YI = np.array(range(I.shape[1]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
XI = np.array(range(I.shape[2]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
extentI = STalign.extent_from_x((YI,XI))

# 5. We can also see the data to be aligned with pandas ------
# Single cell data to be aligned
fname = path + 'Visium/outs/spatial/tissue_positions_list.csv'
df = pd.read_csv(fname, header=None) # Note here needs to say no header, unlike in the Xenium examples
df.shape # 4992, 6
df.head()
df.columns = ["barcodes", "in_tissue", "row", "col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
df.head()
df = df.set_index('barcodes', drop=False)
df.head()
df.shape # 4992, 6

# get cell centroid coordinates --------------------
xM = np.array(df['pxl_row_in_fullres']/7)
yM = np.array(df['pxl_col_in_fullres']/7)

# read in Visium count matrix to get library size
visadata = scanpy.read_10x_h5(path + 'Visium/outs/filtered_feature_bc_matrix.h5')
visadata_barcode = np.array(visadata.obs_names)
colData_barcode = np.array(df.index)
all(visadata_barcode == colData_barcode) # FALSE
visadata_barcode[0] # 'AACACCTACTATCGAA-1'
colData_barcode[0] # 'GTCACTTCCTTCTAGA-1'

set(visadata_barcode).issubset(set(colData_barcode)) # True
set(colData_barcode).issubset(set(visadata_barcode)) # True

# Should sort anndata row sum by the coldata df
visadata.X.sum(axis = 1).shape # row-wise sum
libsize = np.array(visadata.X.sum(axis = 1))
libsizeM = pd.DataFrame(visadata.X.sum(axis = 1), index=visadata_barcode)
libsizeM['barcode'] = libsizeM.index

# libsizeM.barcode = libsizeM.barcode.astype("category")
sorter = df.index.to_list()
libsizeM.barcode = libsizeM.barcode.astype("category").cat.set_categories(sorter)

libsizeMtest = libsizeM.sort_values(["barcode"]) # Wooo Finally, why is Python so complicated just to sort?
libsizeMtest.columns = ["val", "barcode"]
libsizeMplot = np.array(libsizeMtest['val'])

# plot
fig,ax = plt.subplots()
ax.scatter(xM,yM,s=30,alpha=1,c=libsizeMplot, cmap = "gray")
#ax.invert_yaxis()
# ax.invert_xaxis()
plt.show()

# # Normalize
# grey_scaled_libsize = libsizeMplot/libsizeMplot.ptp()
# grey_scaled_libsize = grey_scaled_libsize - grey_scaled_libsize.min()
# grey_scaled_libsize.ptp()
#
# xM, yM, grey_scaled_libsize

## More plotting --------------------------------
# plot
fig, ax = plt.subplots()
ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(xM,yM,s=30,alpha=0.5, c=libsizeMplot)
plt.show()

# ax.get_ybound()
# (-0.5, 2304.625)
# ax.get_xbound()
# (-0.5, 2758.5)

# # Do not rasterize, and show the original image
# XJ = xM # Direct mapping, so no need to rasterize and downsize # len(XJ) 4992
# YJ = yM # Direct mapping, so no need to rasterize and downsize # len(XJ) 4992
# # M = Do not need for downstream, unless for constructing J
# # fig = Do not need for the plotting, just plot the scatter from previous
XJ,YJ,M,fig = STalign.rasterize(x=xM, y=yM, g=libsizeMplot, dx=3)
# TODO: here even I add g=libsizeMplot, the rasterization does not work well on Visium
# ## TODO: why the rasterize here changes the scaling. Maybe it was built for Xenium?
# # len(XJ) = 646
# # len(YJ) = 613
# # M.shape # (1, 613, 646)
# # # M.ptp() # 0.04166618344513297  # 0.0409358862545886  # 0.041051532527712584
#
ax = fig.axes[0]
# ax.invert_yaxis() # Somehow no need here, maybe because I supplied the libsize vector
plt.show()

##################################
print(M.shape)
J = np.vstack((M, M, M)) # make into 3xNxM
print(J.min())
print(J.max())

# normalize
J = STalign.normalize(J)
print(J.min())
print(J.max())

# double check size of things
print(I.shape)
print(M.shape)
print(J.shape)

##################################
# manually make corresponding points
pointsI = np.array([[1050.,950.], [700., 2200.], [500., 1550.], [1550., 1840.]])
pointsJ = np.array([[1500.,1100.], [1200.,2350.], [1000., 1700.], [2000., 1950.]])

# plot
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI)
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)

ax[0].scatter(pointsI[:,1],pointsI[:,0], c='red')
ax[1].scatter(pointsJ[:,1],pointsJ[:,0], c='red')
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red')
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red')

# invert only rasterized image
# ax[1].invert_yaxis()
plt.show()

# fig,ax = plt.subplots()
# ax.imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)
# # ax.invert_yaxis()
# plt.show()

# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsI,pointsJ)

#############################################
print(L)
print(T)
print(L.shape)
print(T.shape)

# note points are as y,x
affine = np.dot(np.linalg.inv(L), [yM - T[0], xM - T[1]])
print(affine.shape)
xMaffine = affine[0,:]
yMaffine = affine[1,:]

# plot
fig,ax = plt.subplots()
ax.scatter(yMaffine,xMaffine,s=30,alpha=0.5, c=libsizeMplot)
ax.imshow((I).transpose(1,2,0))
plt.show()

########### Nonlinear transform #########
# %%time

# run LDDMM
# specify device (default device for STalign.LDDMM is cpu)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# keep all other parameters default
params = {'L':L,'T':T,
          'niter':2000,
          'pointsI':pointsI,
          'pointsJ':pointsJ,
          'device':device,
          'sigmaM':0.15,
          'sigmaB':0.10,
          'sigmaA':0.11,
          'epV': 10,
          'muB': torch.tensor([0,0,0]), # black is background in target
          'muA': torch.tensor([1,1,1]) # use white as artifact
          }

out = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)

# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']

# now transform the points
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_atlas(xv,v,A,[YJ,XJ],J,[YI,XI])
phiipointsJ = STalign.transform_points_target_to_atlas(xv,v,A,pointsJ)

# plot
fig,ax = plt.subplots()

levels = np.arange(-50000,50000,500)

ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to atlas')

ax.imshow(phiiJ.permute(1,2,0),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="m")
plt.show()


############### Apply transform #########
# Finally, we can apply our transform to the original sets of single cell
# centroid positions to achieve their new aligned positions.

# Now apply to points
tpointsI = STalign.transform_points_target_to_atlas(xv,v,A,np.stack([yM, xM], -1))
phiipointsJ = STalign.transform_points_target_to_atlas(xv,v,A,pointsJ)

# plot
fig,ax = plt.subplots()
ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="r")
ax.scatter(tpointsI[:,1].detach(),tpointsI[:,0].detach(),s=30,alpha=0.5,
           c=libsizeMplot)
plt.show()

# save results by appending
new_coord = pd.DataFrame(tpointsI.numpy())
new_coord = new_coord.set_index(df.barcodes)
new_coord.columns = ["pxl_row_in_fullres_transformed",
                     "pxl_col_in_fullres_transformed"]
results = df.join(new_coord)
fname_new = path + 'Visium/outs/spatial/visCoord_to_XeHE_STalign_nonlinear.csv'
results.to_csv(fname_new)





############## Random plotting ##############
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show(block=True)


