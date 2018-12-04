#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile)

from frangi import frangi_from_image
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os
from scoring import confusion, mcc

import matplotlib as mpl

from skimage.segmentation import random_walker


INTERACTIVE = True
filenames = list_by_quality('good')
RW_BETA = 10

for filename in filenames:
    print('running rw_demo on', filename)

    cimg = open_typefile(filename, 'raw')
    ctrace = open_typefile(filename, 'ctrace')
    trace = open_tracefile(filename)
    img = get_named_placenta(filename)
    crop = cropped_args(img)

    if img[crop].shape[0] > img[crop].shape[1]:
        # make the size of figures more consistent
        cimg = np.rot90(cimg)
        ctrace = np.rot90(ctrace)
        trace = np.rot90(trace)
        img = np.rot90(img)
        crop = cropped_args(img)


    plt.close('all')

    cm = mpl.cm.plasma
    cmscales = mpl.cm.magma
    cm.set_bad('k', 1)  # masked areas are black, not white
    cmscales.set_bad('w', 1)

    scales =np.logspace(-1.5, 3.5, num=12, base=2)
    threshold = .4

    W = np.zeros((len(scales), *img.shape), dtype=np.bool)
    WL = np.zeros((len(scales), *img.shape), dtype=np.bool)
    V = np.zeros((len(scales), *img.shape))

    #fig, ax = plt.subplots(ncols=3, nrows=len(scales), figsize=(10,10))

    for n, sigma in enumerate(scales):

        f = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=20,
                            beta=0.35)
        V[n] = f

        f[f == 0] = ma.masked

        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20,6))

        ax[0].imshow(f[crop], cmap=cm)
        ax[0].axis('off')
        ax[0].set_title(r'$V_\sigma,  \sigma={:.3f}$'.format(sigma))

        markers = np.zeros(img.shape, np.int32)
        markers[f.mask] = 1
        markers[f > threshold] = 2

        ax[1].imshow(markers[crop], cmap=plt.cm.viridis, vmin=0, vmax=3)
        ax[1].axis('off')
        ax[1].set_title('markers')

        #rw = random_walker(f.filled(0), markers, beta=RW_BETA)
        rw = random_walker(1-f.filled(0), markers, beta=RW_BETA)
        rw_loose = random_walker(f.filled(0) > 0, markers, beta=RW_BETA)
        W[n] = (rw == 2)
        WL[n] = (rw_loose == 2)

        # set the new stuff to a higher number so you can see what was added
        show_added = rw.copy()
        show_added_loose = rw_loose.copy()
        show_added[~(markers == 2) & (rw==2)] = 3
        show_added_loose[~(markers == 2) & (rw_loose==2)] = 3
        # set the zero stuff back to 0 so you can tell what wasn't filled
        show_added[(rw == 1) & (markers == 0)] = 0
        show_added_loose[(rw_loose == 1) & (markers == 0)] = 0

        ax[2].imshow(show_added[crop], vmin=0, vmax=3)
        ax[2].axis('off')
        ax[2].set_title('RW')
        ax[3].imshow(show_added_loose[crop], vmin=0, vmax=3)
        ax[3].axis('off')
        ax[3].set_title('RW (loose)')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.01)

        if INTERACTIVE:
            plt.show()
        else:
            plt.savefig(f'demo_output/rw_demo/rw_demo_scale_{n:0{2}}.png')
        plt.close('all')

    Vmax, Vargmax = V.max(axis=0), V.argmax(axis=0)
    Vmax = ma.masked_where(Vmax==0, Vmax)
    Vargmax = ma.masked_where(~trace, Vargmax)

    # get the smallest label that matched

    labs = np.argmax(W, axis=0) # returns the first index of boolean
    labs =ma.masked_where(labs==0, labs)
    approx = labs.filled(0)!=0


    labsL = np.argmax(WL, axis=0) # returns the first index of boolean
    labsL =ma.masked_where(labsL==0, labsL)
    approxL = labsL.filled(0)!=0

    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20,12))

    ax[0,0].imshow(cimg[crop])
    ax[0,0].axis('off')
    ax[1,0].imshow(ctrace[crop])
    ax[1,0].axis('off')

    ax[0,1].imshow(Vmax[crop], cmap=cm)
    ax[0,1].axis('off')
    ax[0,1].set_title('$max(V_\sigma)$')
    ax[1,1].imshow((Vargmax)[crop], cmap=cmscales)
    ax[1,1].axis('off')

    ax[0,2].imshow(labs[crop], cmap='magma')
    ax[0,2].axis('off')
    ax[1,2].imshow(confusion(approx, trace, bg_mask=img.mask)[crop])
    ax[1,2].axis('off')

    ax[0,3].imshow(labsL[crop], cmap='magma')
    ax[0,3].axis('off')
    ax[1,3].imshow(confusion(approxL, trace, bg_mask=img.mask)[crop])
    ax[1,3].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.01)

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])
    m, counts = mcc(approx, trace, bg_mask=img.mask, return_counts=True)
    mL, countsL = mcc(approxL, trace, bg_mask=img.mask, return_counts=True)
    p = precision_score(counts)
    pL = precision_score(countsL)

    ax[1,2].set_title(f'MCC: {m:.2f}', loc='right')
    ax[1,2].set_title(f'precision: {p:.2%}')
    ax[1,3].set_title(f'MCC: {mL:.2f}', loc='right')
    ax[1,3].set_title(f'precision: {pL:.2%}')

    if not INTERACTIVE:
        plt.imsave('demo_output/rw_demo/rw_demo_labels.png', labs[crop],
                cmap=cmscales)
    else:
        #plt.imshow(labs[crop], cmap=cmscales)

        plt.show()
