# Below code does not work because the raw image is too large, if you want to normalize, use plugins in Fiji
# Here can't even read it in with PIL python package.
# In STalign tutorial, STalign.normalize() works on the Fiji 0.1 scaled raw tif (1.7GB) image (now 22MB)

# Image norm attempt
# Tried to load the raw tiff in Fiji, wanted to normalize using Plugins > Process > Quantile Based Normalization
# * However, this plugin only takes 8-bit (grey or color_256) image, so had to convert using Image > Type > 8-bit, and saved as png
# Then again tried using that plugin to normalize, and encountered java nullpointexception error.
# Anyway, we don't want to have grey-scale image for feature extraction input.

# Following the previous point *, also tried to convert to a colored png by using Image > Type > 8-bit color.
# Then again tried using that plugin to normalize, and same java nullpointexception error.

# So no image normalization methods found for the raw size file, yet. :D

# # import dependencies ---------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch
# from PIL import Image
# import sys
# sys.path.append("..")
# import STalign
#
# from STalign import STalign
# import scipy.misc
#
# # make plots bigger
# plt.rcParams["figure.figsize"] = (12,10)
# path = "/Users/estelladong/Desktop/raw_data_must/"
#
# image_file = path + 'Xenium/outs/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif'
# save_path = path + 'Xenium/outs/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image_normed.png'
#
# # image_file = path + 'Visium/outs/spatial/tissue_image.tif'
# # save_path = path + 'Visium/outs/spatial/tissue_image_normed.tif'
#
# V = plt.imread(image_file)
# Inorm = STalign.normalize(V)
#
# # fig,ax = plt.subplots()
# # ax.imshow(Inorm)
# # plt.show()
# scipy.misc.imsave(save_path, Inorm)
#
# im = Image.fromarray(Inorm)
# im.show()
# im.save(save_path)
#
# def saveNormImage(image_file, save_path):
#     V = plt.imread(image_file)
#     Inorm = STalign.normalize(V)
#     im = Image.fromarray(Inorm)
#     im.save(save_path)
#
# saveNormImage(image_file, save_path)
#
#     im = Image.fromarray(Inorm)
#     im.save(save_path)
#
# saveNormImage(image_file, save_path)

