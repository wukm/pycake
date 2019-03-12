"""
visual demonstration of the angle of principal directions in a placental
sample at a specified scale
"""

#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage.io import imread
from skimage.util import img_as_float

from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float)

from frangi import frangi_from_image
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

from diffgeo import principal_curvatures, principal_directions


filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)

sigma = 1.5
img = mimg_as_float(img)

print('calculating frangi filter')

f = frangi_from_image(img, sigma=1.5, dark_bg=False, dilation_radius=20,
                      beta=0.35)

print('calculating hessian again (oops)')
H = fft_hessian(img, sigma=1.5)
print('calculating pd where f > .05')
v1, v2 = principal_directions(img, 1.5, H=H, mask=(f < 0.05))
print('done')
vm = ma.masked_array(v2, mask=f<.05)

# this colormap doesn't have any black in it!
cmap = mpl.cm.hsv
# so set the mask to black
cmap.set_bad(color=(0,0,0), alpha=1)

fig, ax = plt.subplots()
cax = ax.imshow(vm[crop], cmap=cmap, vmin=0, vmax=np.pi)
ax.axis('off')
cbar = fig.colorbar(cax, ticks=[0, np.pi/3, np.pi/2, 2*np.pi/3, np.pi])
cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$',
                         r'$\frac{2\pi}{3}$', r'$\pi$'])

ax.set_title(r'leading (local) principal direction, $\sigma=1.5$')
fig.tight_layout()

plt.show() # save manually with the name pd_demo_uniscale.png
