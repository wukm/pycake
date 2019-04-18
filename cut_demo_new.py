#!/usr/bin/env python3

from itertools import combinations_with_replacement

import numpy as np

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile)
from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os
from time import perf_counter



b=0.5; g=0.5;
sigmas = np.logspace(-2, 2.5, num=12, base=2)
dilation_radius = 5
OUTPUT_DIR = 'demo_output/cut_demo_new'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

filename = 'T-BN2899857.png'
img = get_named_placenta(filename)
cimg = open_typefile(filename, 'raw')
crop = cropped_args(img)
gt_mask = open_typefile(filename, 'mask', mode='L').astype('bool')

if img[crop].shape[0] > img[crop].shape[1]:
    img = np.rot90(img)
    crop = cropped_args(img)
    gt_mask = np.rot90(gt_mask)
    cimg = np.rot90(cimg)

name_stub = filename.rstrip('.png').strip('T-')

# use this slice for the cut location in reference to the uncropped image
inset = np.s_[550:800,450:700]

V = np.array([frangi_from_image(img, sigma, beta=b, gamma=g, dark_bg=False,
                        dilation_radius=dilation_radius, rescale_frangi=True).filled(0)
              for sigma in sigmas])

# frangi filter at this scale using ground truth only
V_gt = np.array([frangi_from_image(ma.masked_array(img.data, mask=gt_mask), sigma,
                         beta=b, gamma=g, dark_bg=False, dilation_radius=dilation_radius,
                         rescale_frangi=True)
                 for sigma in sigmas])

f = V.max(axis=0)
f_gt = V_gt.max(axis=0)

cimg_ws = (cimg*~np.expand_dims(img.mask, axis=-1))[inset]
cimg_gt = (cimg*~np.expand_dims(gt_mask, axis=-1))[inset]

plt.imsave(os.path.join(OUTPUT_DIR, 'cut_demo_new_f_ws.png'), f[inset],
           cmap=plt.cm.nipy_spectral, vmin=0, vmax=1.0)
plt.imsave(os.path.join(OUTPUT_DIR, 'cut_demo_new_f_gt.png'), f_gt[inset],
           cmap=plt.cm.nipy_spectral, vmin=0, vmax=1.0)
plt.imsave(os.path.join(OUTPUT_DIR, 'cut_demo_new_cimg_ws.png'), cimg_ws)
plt.imsave(os.path.join(OUTPUT_DIR, 'cut_demo_new_cimg_gt.png'), cimg_gt)
