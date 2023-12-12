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
path = "/Users/estelladong/Desktop/Vis_MOSAIC_Xe_CHUV_IO/"

# 1. Read H&E -------------------------
# Target is H&E staining image
# image_file = path + 'Xenium/outs/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png'
image_file = path + 'L1_2_vis_tissue_hires_image.png'
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
fname = path + 'L1_1_xe_cells.csv.gz'
df = pd.read_csv(fname)
df.head()

# get cell centroid coordinates --------------------
xM = np.array(df['x_centroid'])
yM = np.array(df['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xM,yM,s=1,alpha=0.2)
plt.show()

## More plotting --------------------------------
# plot
fig, ax = plt.subplots()
ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(xM,yM,s=1,alpha=0.1)
plt.show()

# rasterize at 30um resolution (assuming positions are in um units) and plot
XJ,YJ,M,fig = STalign.rasterize(xM, yM, dx=30)

ax = fig.axes[0]
ax.invert_yaxis()
plt.show()

#######################################################
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

#############################################
# plot
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI)
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)
#plt.show()

# manually make corresponding points
pointsI = np.array([[1020., 255.], [680.,   580.], [900.,  850.], [1150.,  580.]])
pointsJ = np.array([[4800.,1800.], [2600., 4400.], [725., 3100.], [2400., 1100.]])

ax[0].scatter(pointsI[:,1],pointsI[:,0], c='red')
ax[1].scatter(pointsJ[:,1],pointsJ[:,0], c='red')
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red')
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red')

# invert only rasterized image - why?
ax[1].invert_yaxis()
plt.show()


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
ax.scatter(yMaffine,xMaffine,s=1,alpha=0.1)
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

##################### Get the grid #####################
# now transform the points
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_source(xv,v,A,[YJ,XJ],J,[YI,XI])
phiipointsJ = STalign.transform_points_target_to_source(xv,v,A,pointsJ)

#switch tensor from cuda to cpu for plotting with numpy
if phi.is_cuda:
    phi = phi.cpu()
if phiiJ.is_cuda:
    phiiJ = phiiJ.cpu()
if phiipointsJ.is_cuda:
    phiipointsJ = phiipointsJ.cpu()

# plot
fig,ax = plt.subplots()

levels = np.arange(-50000,50000,500)

ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to source')

ax.imshow(phiiJ.permute(1,2,0),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="m")

############### Apply transform #########
# Finally, we can apply our transform to the original sets of single cell
# centroid positions to achieve their new aligned positions.

# Now apply to points
tpointsJ = STalign.transform_points_target_to_source(xv,v,A,np.stack([yM, xM], -1))

#switch tensor from cuda to cpu for plotting with numpy
if tpointsJ.is_cuda:
    tpointsJ = tpointsJ.cpu()

# plot
fig,ax = plt.subplots()
ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="g")
ax.scatter(pointsI[:,1],pointsI[:,0], c='r')
ax.scatter(tpointsJ[:,1].detach(),tpointsJ[:,0].detach(),s=1,alpha=0.1)
plt.show()

# save results by appending
# results = np.hstack((df, tpointsJ.numpy()))
fname_new = path + 'L1_1_xe_cells_STalign.csv'

# Only affine result
dict_of_arrs = {"x_centroid_transform": tpointsJ.numpy()[:, 0],
                "y_centroid_transform": tpointsJ.numpy()[:, 1]}
new_coord = pd.DataFrame(dict_of_arrs)
results = df.join(new_coord)
results = results.set_index('cell_id')

results.to_csv(fname_new)



