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
from time import perf_counter



b=0.5; g=0.5;
scales = np.logspace(-1, 4, num=25, base=2)

W_times = list()
V_times = list()

placentas = list_by_quality(0,N=1)
OUTPUT_DIR = 'demo_output/weingarten_demo_quality_0-190411'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for filename in placentas:

    img = get_named_placenta(filename)
    crop = cropped_args(img)

    if img[crop].shape[0] > img[crop].shape[1]:
        img = np.rot90(img)
        crop = cropped_args(img)

    name_stub = filename.rstrip('.png').strip('T-')

    start = perf_counter()
    V = np.array([frangi_from_image(img, sigma, beta=b, gamma=g, dark_bg=False,
                            dilation_radius=20,
                            rescale_frangi=True)[crop].filled(0)
                for sigma in scales])

    V_elapsed = perf_counter() - start
    V_times.append(V_elapsed)

    start = perf_counter()

    W = np.array([frangi_from_image(img, sigma, beta=b, gamma=g, dark_bg=False,
                            dilation_radius=20, rescale_frangi=True,
                            use_real_weingarten_map=True)[crop].filled(0)
                for sigma in scales])

    W_elapsed = perf_counter() - start
    W_times.append(W_elapsed)

    Vmax = V.max(axis=0)
    Wmax = W.max(axis=0)

    plt.cm.rainbow.set_bad('k')

    mask_under = lambda F, a: ma.masked_array(F.argmax(axis=0),
                                            mask=F.max(axis=0) < a)
    Vargmax = mask_under(V, .3)
    Wargmax = mask_under(W, .3)

    #fig, ax = plt.subplots(ncols=2, nrows=2)

    #ax[0,0].imshow(Vmax, cmap=plt.cm.nipy_spectral)
    #ax[0,1].imshow(Wmax, cmap=plt.cm.nipy_spectral)
    #ax[1,0].imshow(Vargmax, cmap=plt.cm.rainbow)
    #ax[1,1].imshow(Wargmax, cmap=plt.cm.rainbow)

    #[a.axis('off') for a in ax.ravel()]

    plt.imsave(os.path.join(OUTPUT_DIR, ''.join((name_stub, '-Vmax.png'))),
                            Vmax, cmap=plt.cm.nipy_spectral)
    plt.imsave(os.path.join(OUTPUT_DIR, ''.join((name_stub, '-Wmax.png'))),
               Wmax, cmap=plt.cm.nipy_spectral)
    plt.imsave(os.path.join(OUTPUT_DIR, ''.join((name_stub, '-Vargmax.png'))),
               Vargmax, cmap=plt.cm.rainbow)
    plt.imsave(os.path.join(OUTPUT_DIR, ''.join((name_stub, '-Wargmax.png'))),
               Wargmax, cmap=plt.cm.rainbow)

    print('Finished with sample', name_stub)
    print(f'\tFrangi: {V_elapsed} sec\t', f'Wein: {W_elapsed} sec')
