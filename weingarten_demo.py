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


filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)

def vshow(ar):
    plt.imshow(ar, cmap=plt.cm.nipy_spectral, vmin=0, vmax=1.0)
    plt.show()

b=0.5; g=0.5;
scales = np.logspace(-1, 4, num=20, base=2)

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

Vargmax = ma.masked_array(V.argmax(axis=0), mask=(Vmax<.5))
Wargmax = ma.masked_array(W.argmax(axis=0), mask=(Wmax<.5))
