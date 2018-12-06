# coding: utf-8
#filename = 'T-BN4981652.png'
#img = get_named_placenta(filename)
#crop = cropped_args(img)
#plt.imshow(img)
#s = plt.show
#s()
#plt.imshow(img > .9*img.mask)
#plt.show()
#plt.imshow(img > .9*img.max())
#plt.show()
#plt.imshow(img > .8*img.max())
#plt.show()
#plt.imshow(img > .75*img.max())
#plt.show()
#from skimage.filters import sobel
#sobel(img)
#plt.imshow(_)
#plt.show()
#from skimage import filters
#filters.laplace(img)
#plt.imshow(_)
#plt.show()
#dilate_boundary(img, 20)
#plt.imshow(_)
#plt.show()
#filters.laplace(dilate_boundary(img, 30))
#plt.imshow(_)
#plt.show()
#filters.laplace(dilate_boundary(img, 30) == 0)
#plt.imshow(_)
#s()
#filters.laplace(dilate_boundary(img, 30))==0
#plt.imshow(_)
#plt.show()
#from skimage import morphology as mph
#from frangi import frangi_from_image
#fft_gradient(img, sigma=20)
#plt.imshow(_)
#plt.show()
#fft_gradient(img, sigma=15)
#plt.imshow(_)
#plt.show()
#raw_img
#rimg
from placenta import open_typefile
open_typefile(filename, 'raw')
plt.imshow(_)
plt.show()
raw = open_typefile(filename, 'raw')
plt.imshow(raw[...,1])
plt.show()
plt.imshow(fft_gradient(raw[...,1],sigma=15) )
plt.show()
plt.imshow(fft_gradient(raw[...,1],sigma=.01))
plt.show()
plt.imshow(fft_gradient(raw[...,1],sigma=.001))
plt.show()
from skimage.segmentation import watershed
get_ipython().run_line_magic('pinfo', 'watershed')
marks = np.zeros(img.shape, np.int32)
marks[0,0] = 1
marks[img.shape//2]
marks[img.shape[0]//2,img.shape[1]//2]
marks[img.shape[0]//2,img.shape[1]//2] = 2
watershed(fft_gradient(raw[...,1],sigma=.01))
watershed(fft_gradient(raw[...,1],sigma=.01), marks)
plt.imshow(_)
plt.show()
watershed(fft_gradient(raw[...,1],sigma=.001), marks)
plt.show()
plt.imshow(_)
plt.show()
g = fft_gradient(raw[...,1],sigma=.01)
plt.imshow(g)
plt.show()
g = fft_gradient(raw[...,1],sigma=.001)
plt.imshow(g)
plt.show()
from skimage.segmentation import find_boundaries
find_boundaries(g)
plt.imshow(_)
plt.show()
plt.imshow(g>g.mean())
plt.show()
marks[g>g.mean()] = 2
watershed(g, marks)
plt.imshow(_)
plt.show()
from scipy.ndimage import find_objects
find_objects(watershed(g,marks))
help(find_objects)
from scipy.ndimage import label
get_ipython().run_line_magic('pinfo', 'label')
label(watershed(g,marks))
help(label)
plt.imshow(watershed(g,marks))
plt.show()
plt.imshow(watershed(g,marks))
plt.show()
from skimage.morphology import binary_erosion
plt.imshow(watershed(g,marks), vmin=0, vmax=2)
plt.show()
w = watershed(g,marks)
plt.imshow(w)
plt.show()
from skimage.morphology import binary_erosion, disk
binary_erosion(w, disk(10))
plt.imshow(_)
plt.show()
plt.imshow(w-1)
plt.show()
plt.imshow(binary_erosion(w-1, disk(10)))
plt.show()
# oops need to make watershed thing binary. w-1 is hacky but works
eroded = binary_erosion(w-1, disk(10))
find_boundaries(eroded)
plt.imshow(_)
plt.show()
import scipy.ndimage as ndi
ndi.label(eroded)
plt.imshow(_[0])
plt.show()
labs, nlabs = ndi.label(eroded)
for i in range(nlabs):
    print(i, np.sum(labs==i))

labs==1
plt.imshow(_)
plt.show()
plt.imshow(np.logical_xor(labs==1, ~img.mask))
plt.show()
sum(~img.mask)
np.sum(~img.mask)
np.sum(labs==1)
