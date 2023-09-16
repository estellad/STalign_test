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
#
# fig,ax = plt.subplots()
# ax.imshow(Inorm)
#
# #







