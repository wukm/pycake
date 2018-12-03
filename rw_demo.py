#!/usr/bin/env python3

"""
THIS IS A BASIC SET UP TO EXPLORE, PLAY AROUND A BIT
"""
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float)

from frangi import frangi_from_image
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

import matplotlib as mpl

from skimage.segmentation import random_walker

filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)

cm = mpl.cm.plasma
cmscales = mpl.cm.viridis
cm.set_bad('k', 1)  # masked areas are black, not white
cmscales.set_bad('k', 1)
#scales = [0.5, 1.5, 3.0, 6.0]
scales =np.linspace(0.2, 6, num=10)
W = np.zeros((len(scales), *img.shape), dtype=np.bool)

for n, sigma in enumerate(scales):
    f = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=20,
                           beta=0.35)

    f[f == 0] = ma.masked

    plt.imshow(f[crop], cmap=cm)
    plt.axis('off')
    plt.title(r'Frangi $\sigma={:.2f}$ '.format(sigma) +
                  'zero-masked for random-walker')
    plt.colorbar(shrink=0.5)
    plt.show()  # adjust size manually

    # markers for random walker
    markers = np.zeros(img.shape, np.int32)
    markers[f.mask] = 1
    markers[f > .4] = 2

    plt.imshow(markers[crop], cmap=plt.cm.viridis)
    plt.axis('off')
    plt.show()

    rw = random_walker(f.filled(0), markers)
    # set the new stuff to a higher number so you can see what was added
    rw[~(markers == 2) & (rw==2)] = 3
    # set the zero stuff back to 0 so you can tell what wasn't filled
    rw[(rw == 1) & (markers == 0)] = 0
    plt.imshow(rw[crop])
    plt.axis('off')
    plt.show()

    W[n] = (rw >= 2)

# get the smallest label that matched

labs = np.argmax(W, axis=0) # returns the first index of boolean
labs =ma.masked_where(labs==0, labs)
plt.imshow(labs[crop], cmap=cmscales)
plt.axis('off')
plt.show()
