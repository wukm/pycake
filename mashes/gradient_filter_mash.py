# coding: utf-8
get_ipython().run_line_magic('clear', '')
from placenta import get_named_placenta, list_by_quality
list_by_quality(N=1)
list_by_quality(0)
filename = list_by_quality(N=1)[0]
filename
from frangi import frangi_from_image
from hfft import fft_gradient
get_ipython().run_line_magic('pinfo', 'fft_gradient')
img = get_named_placenta(filename)
img
import numpy as np
import matplotlib.pyplot as plt
fft_gradient(img, sigma=2.0)
G = _
plt.imshow(fft_gradient)
G.max()
G.min()
from skimage.util import img_as_flat
from skimage.util import img_as_float
img_as_float(G)
plt.imshow(_)
plt.show()
from plate_morphology import dilate_boundary
dilate_boundary(G, mask=img.mask, radius=20)
plt.imshow(_.filled(0))
plt.show()
G = dilate_boundary(G, mask=img.mask, radius=20).filled(0)
plt.imshow(G)
plt.show()
plt.imshow(np.isclose(G,0))
plt.show()
from merging import nz_percentile
nz_percentile(G,5)
plt.imshow(G < _)
plt.show()
nz_percentile(G,50)
plt.imshow(G < _)
plt.show()
frangi_from_image(img, 1, dark_bg=False, beta=0.35)
frangi_from_image(np.ma.masked_array(img_as_float(img), mask=img.mask), 1, dark_bg=False, beta=0.35)
plt.imshow(_)
plt.show()
frangi_from_image(np.ma.masked_array(img_as_float(img), mask=img.mask), 1, dark_bg=False, beta=0.35)
plt.imshow(_.filled(0))
plt.show()
frangi_from_image(np.ma.masked_array(img_as_float(img), mask=img.mask), 1, dark_bg=False, beta=0.5, dilation_radius=20)
F = _
plt.imshow(_.filled(0))
plt.show()
plt.imshow(G*F.filled(0))
plt.show()
F
G
low_G = G < nz_percentile(G, 50)
plt.imshow(low_G)
plt.show()
plt.imshow(low_G*F.filled(0))
plt.show()
plt.imshow(low_G*F.filled(0), vmin=0, vmax=1)
plt.show()
fig, ax = plt.subplots(ncols=3, nrows=1)
axes = ax.ravel()
axes[0]
from placenta import cropped_args
cropped_args(img)
crop = _
axes[0] = plt.imshow(F.filled(0)[crop], vmin=0, vmax=1)
axes[1] = plt.imshow(low_G*F.filled(0)[crop], vmin=0, vmax=1)
axes[1] = plt.imshow((low_G*F.filled(0))[crop], vmin=0, vmax=1)
axes[2] = plt.imshow(low_G[crop], cmap='gray_r')
plt.show()
plt.imshow((low_G*F.filled(0))[crop])
plt.show()
fig, ax = plt.subplots(ncols=3, nrows=1)
ax[0][0] = plt.imshow(F.filled(0)[crop], vmin=0, vmax=1)
ax[0][0].imshow(F.filled(0)[crop], vmin=0, vmax=1)
ax[0,0].imshow(F.filled(0)[crop], vmin=0, vmax=1)
ax[0].imshow(F.filled(0)[crop], vmin=0, vmax=1)
ax[1].imshow((low_G*F.filled(0))[crop], vmin=0, vmax=1)
ax[2].imshow(low_G[crop], cmap='gray', vmin=0, vmax=1)
plt.show()
