from PIL import Image

img_path = r"path\to\image.tif"

img = Image.open(img_path)

# get the image data
data = img.getdata()

# save the RGB image
outpath = img_path[:-4] + '_rgb.png'
img.save(outpath)