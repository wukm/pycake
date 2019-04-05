#!/usr/bin/env python3

from itertools import combinations_with_replacement

import numpy as np

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os


filename = list_by_quality(N=2)[1]
img = get_named_placenta(filename)
crop = cropped_args(img)

def vshow(ar):
    plt.imshow(ar, cmap=plt.cm.nipy_spectral, vmin=0, vmax=1.0)
    plt.show()

b=0.35; g=0.5;
scales = np.logspace(-1, 4, num=12, base=2)

V = np.array([frangi_from_image(img, sigma, beta=b, gamma=g, dark_bg=False,
                        dilation_radius=20,
                        rescale_frangi=True)[crop].filled(0)
             for sigma in scales])

W = np.array([frangi_from_image(img, sigma, beta=b, gamma=g, dark_bg=False,
                        dilation_radius=20, rescale_frangi=True,
                        use_real_weingarten_map=True)[crop].filled(0)
             for sigma in scales])

Vmax = V.max(axis=0)
Wmax = W.max(axis=0)

plt.cm.rainbow.set_bad('k')

mask_under = lambda F, a: ma.masked_array(F.argmax(axis=0),
                                          mask=F.max(axis=0) < a)
Vargmax = mask_under(V, .4)
Wargmax = mask_under(W, .4)

fig, ax = plt.subplots(ncols=2, nrows=2)

ax[0,0].imshow(Vmax, cmap=plt.cm.nipy_spectral)
ax[0,1].imshow(Wmax, cmap=plt.cm.nipy_spectral)
ax[1,0].imshow(Vargmax, cmap=plt.cm.rainbow)
ax[1,1].imshow(Wargmax, cmap=plt.cm.rainbow)

[a.axis('off') for a in ax.ravel()]
