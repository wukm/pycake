#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile)

from frangi import frangi_from_image
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

import matplotlib as mpl

from skimage.segmentation import random_walker

INTERACTIVE = True
filename = list_by_quality('good')[0]

print('running rw_demo on', filename)

cimg = open_typefile(filename, 'raw')
img = get_named_placenta(filename)
crop = cropped_args(img)

if INTERACTIVE:
    plt.imsave('demo_output/rw_demo/rw_demo_base.png', cimg[crop])
else:
    plt.show()

plt.close('all')

cm = mpl.cm.plasma
cmscales = mpl.cm.Blues
cm.set_bad('k', 1)  # masked areas are black, not white
cmscales.set_bad('k', 1)

scales =np.logspace(-1.5, 3.5, num=12, base=2)
threshold = .4

W = np.zeros((len(scales), *img.shape), dtype=np.bool)

#fig, ax = plt.subplots(ncols=3, nrows=len(scales), figsize=(10,10))

for n, sigma in enumerate(scales):

    f = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=20,
                           beta=0.35)

    f[f == 0] = ma.masked

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20,6))

    ax[0].imshow(f[crop], cmap=cm)
    ax[0].axis('off')

    markers = np.zeros(img.shape, np.int32)
    markers[f.mask] = 1
    markers[f > threshold] = 2

    ax[1].imshow(markers[crop], cmap=plt.cm.viridis, vmin=0, vmax=3)
    ax[1].axis('off')
    ax[1].set_title(r'$\sigma={:.3f}$'.format(sigma))

    rw = random_walker(f.filled(0), markers, beta=20)
    W[n] = (rw == 2)

    # set the new stuff to a higher number so you can see what was added
    show_added = rw.copy()
    show_added[~(markers == 2) & (rw==2)] = 3
    # set the zero stuff back to 0 so you can tell what wasn't filled
    show_added[(rw == 1) & (markers == 0)] = 0

    ax[2].imshow(show_added[crop], vmin=0, vmax=3)
    ax[2].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.01)

    if INTERACTIVE:
        plt.show()
    else:
        plt.savefig(f'demo_output/rw_demo/rw_demo_scale_{n:0{2}}.png')
    plt.close('all')


# get the smallest label that matched


labs = np.argmax(W, axis=0) # returns the first index of boolean
labs =ma.masked_where(labs==0, labs)

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(labs[crop], cmap=cmscales)
plt.axis('off')
# i should make this uniform with the rest but it looks like trash for some
# reason

if INTERACTIVE:
    plt.imsave('demo_output/rw_demo/rw_demo_labels.png', labs[crop],
               cmap=cmscales)
else:
    plt.show()
