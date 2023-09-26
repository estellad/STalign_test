# https://jef.works/STalign/notebooks/xenium-heimage-alignment.html
# import dependencies ---------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from STalign import STalign

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)
path = "/Users/estelladong/Desktop/raw_data_must"

# 1. Read data -------------------------
# Single cell data 1
# read in data
fname = path + '/Xenium/outs/cells.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())

# 2. Xe1 plot --------------------------
# get cell centroid coordinates
xI = np.array(df1['x_centroid'])
yI = np.array(df1['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)
plt.show()

# 3. Rasterize -------------------------
XI,YI,I,fig = STalign.rasterize(xI,yI,dx=30)

# plot
ax = fig.axes[0]
ax.invert_yaxis()
plt.show()


# 4. Repeat 1-3 for second dataset -----
# Single cell data 2
# read in data
fname = path + '/Xenium2/outs/cells.csv.gz'
df2 = pd.read_csv(fname)

# get cell centroids
xJ = np.array(df2['x_centroid'])
yJ = np.array(df2['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2,c='#ff7f0e')
plt.show()

# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=30)
ax = fig.axes[0]
ax.invert_yaxis()
plt.show()

# 5. Plot overlay of unaligned ------------
# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)
ax.scatter(xJ,yJ,s=1,alpha=0.2)
plt.show()

# 6. Save .npz files ----------------------
# Optional: write out npz files for landmark point picker
np.savez(path + '/Xenium/outs/Xenium_Breast_Cancer_Rep1', x=XI,y=YI,I=I)
np.savez(path + '/Xenium2/outs/Xenium_Breast_Cancer_Rep2', x=XJ,y=YJ,I=J)
# outputs Xenium_Breast_Cancer_Rep1.npz and Xenium_Breast_Cancer_Rep2.npz

# 7. Run the command line to get GUI to select landmark ----------
# Put https://github.com/JEFworks-Lab/STalign/blob/main/STalign/curve_annotator.py
# to current project folder.
# python curve_annotator.py '/Users/estelladong/Desktop/raw_data_must/Xenium/outs/Xenium_Breast_Cancer_Rep1.npz'
# python curve_annotator.py '/Users/estelladong/Desktop/raw_data_must/Xenium2/outs/Xenium_Breast_Cancer_Rep2.npz'

# 8. Read landmarks selected from GUI ---------------------------
# read from file
pointsIlist = np.load(path + '/Xenium/outs/Xenium_Breast_Cancer_Rep1_curves.npy', allow_pickle=True).tolist()
print(pointsIlist)
# {'0': [(2131.810605092817, 3305.2189819028013)], '1': [(6547.254153479913, 4373.606078676995)], '2': [(4185.11705670572, 5007.960917386672)]}
pointsJlist = np.load(path + '/Xenium2/outs/Xenium_Breast_Cancer_Rep2_curves.npy', allow_pickle=True).tolist()
print(pointsJlist)
# {'A': [(2390.89730782228, 1485.9550435128795)], 'B': [(6831.381178790022, 2445.834075770944)], 'C': [(4544.365049757764, 3046.801817706428)]}

# 9. Convert to array ---------------------------
# convert to array
pointsI = []
pointsJ = []

# Jean's note: a bit odd to me that the points are stored as y,x
## instead of x,y but all downstream code uses this orientation
for i in pointsIlist.keys():
    pointsI.append([pointsIlist[i][0][1], pointsIlist[i][0][0]])
for i in pointsJlist.keys():
    pointsJ.append([pointsJlist[i][0][1], pointsJlist[i][0][0]])

pointsI = np.array(pointsI)
# array([[3305.2189819 , 2131.81060509],
#        [4373.60607868, 6547.25415348],
#        [5007.96091739, 4185.11705671]])
pointsJ = np.array(pointsJ)
# array([[1485.95504351, 2390.89730782],
#        [2445.83407577, 6831.38117879],
#        [3046.80181771, 4544.36504976]])

# 10. Plot them on image
# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) # just want 201x276 matrix
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) # just want 201x276 matrix
# with points
ax[0].scatter(pointsI[:,1], pointsI[:,0], c='red')
ax[1].scatter(pointsJ[:,1], pointsJ[:,0], c='red')
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red')
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red')
ax[0].invert_yaxis()
ax[1].invert_yaxis()
plt.show()

# 11. Affine ---------------------------------------
# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsI, pointsJ)

# 12. Run non-linear model -------------------------
# %%time

# run LDDMM
# specify device (default device for STalign.LDDMM is cpu)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# keep all other parameters default
params = {'L':L,'T':T,
          'niter':200,
          'pointsI':pointsI,
          'pointsJ':pointsJ,
          'device':device,
          'sigmaM':1.5,
          'sigmaB':1.0,
          'sigmaA':1.5,
          'epV': 100
          }

Ifoo = np.vstack((I, I, I)) # make RGB instead of greyscale
Jfoo = np.vstack((J, J, J)) # make RGB instead of greyscale
out = STalign.LDDMM([YI,XI],Ifoo,[YJ,XJ],Jfoo,**params)
plt.show()

# 13. Get non-linear output vars ------------------------
# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']

# 15. Get Transformation grid (PointI for Xe1) ---------------------------
# apply transform
phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
phiI = STalign.transform_image_atlas_to_target(xv,v,A,[YI,XI],Ifoo,[YJ,XJ])
phipointsI = STalign.transform_points_atlas_to_target(xv,v,A,pointsI)

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XJ,YJ,phii[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XJ,YJ,phii[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('source to target')
ax.imshow(phiI.permute(1,2,0)/torch.max(phiI),extent=extentJ)
ax.scatter(phipointsI[:,1].detach(),phipointsI[:,0].detach(),c="m")
ax.invert_yaxis()
plt.show()

# 15. Result is invertible (PointJ for Xe2) ---------------------------
# transform is invertible
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_atlas(xv,v,A,[YJ,XJ],Jfoo,[YI,XI])
phiipointsJ = STalign.transform_points_target_to_atlas(xv,v,A,pointsJ)

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to source')
ax.imshow(phiiJ.permute(1,2,0)/torch.max(phiiJ),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="m")
ax.invert_yaxis()
plt.show()

# 16. Map to original coordinates -----------------------------
# apply transform to original points
tpointsJ = STalign.transform_points_target_to_atlas(xv,v,A, np.stack([yJ, xJ], 1))

# just original points for visualizing later
tpointsI = np.stack([xI, yI])

# 17. Plot overlay --------------------------------------------
# plot results
fig,ax = plt.subplots()
ax.scatter(tpointsI[0,:],tpointsI[1,:],s=1,alpha=0.2)
ax.scatter(tpointsJ[:,1],tpointsJ[:,0],s=1,alpha=0.1) # also needs to plot as y,x not x,y
plt.show()

# 18. Save new coords -----------------------------------------
# save results by appending
# note results are in y,x coordinates
results = np.hstack((df2, tpointsJ.numpy()))
np.savetxt(path + '/Xenium2/outs/Xenium_Breast_Cancer_Rep2_STalign_to_Rep1.csv', results, delimiter=',')

results.shape # (118752, 11)
df2.shape # (118752, 9)
df1.shape # (167780, 9)

resultsdf = pd.DataFrame(results)
print(resultsdf.head()) # two new cols are added to the end for the new Coords