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
WM = out['WM']

# 16. Map to original coordinates -----------------------------
# apply transform to original points (from mouse brain tutorial Jean suggested)
tpointsJ= STalign.transform_points_target_to_atlas(xv,v,A, np.stack([yJ, xJ], 1))

#switch tensor from cuda to cpu for plotting with numpy
if tpointsJ.is_cuda:
    tpointsJ = tpointsJ.cpu()

# switch from row column coordinates (y,x) to (x,y)
xJ_LDDMM = tpointsJ[:, 1]
yJ_LDDMM = tpointsJ[:, 0]

# just Xe1 points for visualizing later
tpointsI = np.stack([xI, yI])

# # apply transform to Xe2 points (from original tutorial)
# tpointsJ = STalign.transform_points_target_to_atlas(xv,v,A, np.stack([yJ, xJ], 1))
#
# # just Xe1 points for visualizing later
# tpointsI = np.stack([xI, yI])

# 17. Plot overlay --------------------------------------------
# plot results
fig,ax = plt.subplots()
ax.scatter(tpointsI[0,:],tpointsI[1,:],s=1,alpha=0.2)
ax.scatter(tpointsJ[:,1],tpointsJ[:,0],s=1,alpha=0.1) # also needs to plot as y,x not x,y
plt.show()

# # 18. Save new coords -----------------------------------------
# # save results by appending
# # note results are in y,x coordinates
# results = np.hstack((df2, tpointsJ.numpy()))
# np.savetxt(path + '/Xenium2/outs/Xenium_Breast_Cancer_Rep2_STalign_to_Rep1.csv', results, delimiter=',')


# 19. Subset to common region -------------------------------------------------------
# compute weight values for transformed source points from target image pixel locations and weight 2D array (matching)
testM = STalign.interp([YI,XI],WM[None].float(),tpointsJ[None].permute(-1,0,1).float())

#switch tensor from cuda to cpu for plotting with numpy
if testM.is_cuda:
    testM = testM.cpu()

fig,ax = plt.subplots()
scatter = ax.scatter(tpointsJ[:,1],tpointsJ[:,0],c=testM[0,0],s=0.1,vmin=0,vmax=1, label='WM values')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="WM values")
plt.show()

# 20. Obtain 'results', save the new aligned positions by appending to our original data ----------------------
if tpointsJ.is_cuda:
    df3 = pd.DataFrame(
        {"aligned_x": xJ_LDDMM.cpu(),
         "aligned_y": yJ_LDDMM.cpu(),
        },
    )
else:
    df3 = pd.DataFrame(
        {"aligned_x": xJ_LDDMM,
         "aligned_y": yJ_LDDMM,
        },
    )
results = pd.concat([df2, df3], axis=1)
results.head()

# 21. save 'weight' values to 'results' --------------------------------
results['WM_values'] = testM[0,0]

# 22. check weight histogram -------------------------------------------
fig,ax = plt.subplots()
ax.hist(results['WM_values'], bins = 20)
plt.show()