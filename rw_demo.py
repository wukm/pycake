#!/usr/bin/env python3

"""
do a visual demonstration of the scalewise random walker method
"""

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile,
                      measure_ncs_markings, add_ucip_to_mask)

from frangi import frangi_from_image
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os
from scoring import confusion, mcc
from preprocessing import inpaint_hybrid

import matplotlib as mpl

from skimage.segmentation import random_walker

import json


def rw_demo(filename, rw_beta, threshold, output_dir=None):


    # ideally this would be a class with all of these
    cimg = open_typefile(filename, 'raw')
    ctrace = open_typefile(filename, 'ctrace')
    trace = open_tracefile(filename)
    img = get_named_placenta(filename)
    crop = cropped_args(img)
    ucip = open_typefile(filename, 'ucip')
    img = inpaint_hybrid(img)

    # make the size of figures more consistent
    if img[crop].shape[0] > img[crop].shape[1]:
        # and rotating it would be fix all this automatically
        cimg = np.rot90(cimg)
        ctrace = np.rot90(ctrace)
        trace = np.rot90(trace)
        img = np.rot90(img)
        ucip = np.rot90(ucip)
        crop = cropped_args(img)

    ucip_midpoint, _ = measure_ncs_markings(ucip)
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=60, mask=img.mask)

    plt.close('all')

    cm = mpl.cm.plasma
    cmscales = mpl.cm.magma
    cm.set_bad('k', 1)  # masked areas are black, not white
    cmscales.set_bad('w', 1)

    scales =np.logspace(-1.5, 3.5, num=12, base=2)

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
        #ax[0].set_title(rf'$V_{{\sigma_{{{n}}}}},\;'
                        #rf'\sigma_{{{n}}}={sigma:.3f}$')
        ax[0].set_title(rf'Frangi score, $(\sigma_{{{n}}}={sigma:.3f})$')

        markers = np.zeros(img.shape, np.int32)
        markers[f.mask] = 1
        markers[f > THRESHOLD] = 2

        ax[1].imshow(markers[crop], cmap=plt.cm.viridis, vmin=0, vmax=3)
        ax[1].axis('off')
        ax[1].set_title('markers')

        rw = random_walker(1-f.filled(0), markers, beta=RW_BETA)
        rw_L = random_walker(f.filled(0) > 0, markers, beta=RW_BETA)
        W[n] = (rw == 2)
        WL[n] = (rw_L == 2)

        # set the new stuff to a higher number so you can see what was added
        show_added = rw.copy()
        show_added_L = rw_L.copy()
        show_added[~(markers == 2) & (rw==2)] = 3
        show_added_L[~(markers == 2) & (rw_L==2)] = 3
        # set the zero stuff back to 0 so you can tell what wasn't filled
        show_added[(rw == 1) & (markers == 0)] = 0
        show_added_L[(rw_L == 1) & (markers == 0)] = 0

        ax[2].imshow(show_added[crop], vmin=0, vmax=3)
        ax[2].axis('off')
        ax[2].set_title('rw')
        ax[3].imshow(show_added_L[crop], vmin=0, vmax=3)
        ax[3].axis('off')
        ax[3].set_title('rw-loose')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.00, wspace=0.01)

        if output_dir is not None:
            fig.savefig(f'./output/{output_dir}/{basename}_{n:{0}2}.png')
        if INTERACTIVE:
            plt.show()

    Vmax, Vargmax = V.max(axis=0), V.argmax(axis=0)
    Vmax = ma.masked_where(Vmax==0, Vmax)
    #Vargmax = ma.masked_where(~trace, Vargmax)

    labs_FA = Vargmax*(Vmax > THRESHOLD).filled(0)
    approx_FA =  labs_FA!=0
    confuse_FA = confusion(approx_FA, trace, bg_mask=ucip_mask)
    # get the smallest label that matched

    labs = np.argmax(W, axis=0) # returns the first index of boolean
    labs =ma.masked_where(labs==0, labs)
    approx = labs.filled(0)!=0
    confuse = confusion(approx, trace, bg_mask=ucip_mask)


    labs_L = np.argmax(WL, axis=0) # returns the first index of boolean
    labs_L =ma.masked_where(labs_L==0, labs_L)
    approx_L = labs_L.filled(0)!=0
    confuse_L = confusion(approx_L, trace, bg_mask=ucip_mask)

    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20,12))

    ax[0,0].imshow(cimg[crop])
    ax[0,0].axis('off')
    ax[0,0].set_title(basename)

    ax[1,0].imshow(ctrace[crop])
    ax[1,0].axis('off')
    ax[1,0].set_title('ground truth')

    ax[0,1].imshow(Vmax[crop], cmap=cm)
    ax[0,1].axis('off')
    ax[0,1].set_title('$\max(V_\sigma)$')


    ax[0,2].imshow(labs[crop], cmap='magma')
    ax[0,2].axis('off')
    ax[0,2].set_title('segmentation (rw)')

    ax[0,3].imshow(labs_L[crop], cmap='magma')
    ax[0,3].axis('off')
    ax[0,3].set_title('segmentation (rw-loose)')

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])

    m, counts = mcc(approx, trace, bg_mask=ucip_mask, return_counts=True)
    m_L, counts_L = mcc(approx_L, trace, bg_mask=ucip_mask, return_counts=True)
    m_FA, counts_FA = mcc(approx_FA, trace, bg_mask=ucip_mask,
                          return_counts=True)

    p = precision_score(counts)
    p_L = precision_score(counts_L)
    p_FA = precision_score(counts_FA)

    ax[1,1].imshow(confuse_FA[crop])
    ax[1,1].axis('off')
    ax[1,1].set_title(rf'   fixed $\alpha={THRESHOLD}$', loc='left')
    ax[1,1].set_title(f'MCC: {m_FA:.2f}\n'
                      f'precision: {p_FA:.2%}', loc='right')
    ax[1,2].imshow(confuse[crop])
    ax[1,2].axis('off')
    ax[1,2].set_title('   scalewise-RW', loc='left')
    ax[1,2].set_title(f'MCC: {m:.2f}\n'
                      f'precision: {p:.2%}', loc='right')

    ax[1,3].imshow(confuse_L[crop])
    ax[1,3].axis('off')
    ax[1,3].set_title('   scalewise-RW (loose)', loc='left')
    ax[1,3].set_title(f'MCC: {m_L:.2f}\n'
                      f'precision: {p_L:.2%}', loc='right')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.01)

    if output_dir is not None:
        fig.savefig(f'./output/{output_dir}/{basename}_m.png')

    return (filename, m_FA, p_FA, m, p, m_L, p_L)



INTERACTIVE = True
filenames = list_by_quality(1)
filenames.extend(list_by_quality(2))
filenames.extend(list_by_quality(3))
filenames.extend(list_by_quality(0))
RW_BETA = 10
THRESHOLD = .4

run_data = list()

for N, filename in enumerate(filenames):


    basename = filename.strip('T-').rstrip('.png')
    print('running rw_demo on', basename, f'({N+1} of {len(filenames)})')
    row = rw_demo(filename, RW_BETA, THRESHOLD, None)
    run_data.append(row)

    if INTERACTIVE:

        plt.show()

    #print(row)

    # do this incrementally; i'm afraid
    if (N % 25 == 0) and (N > 0):
        print('backing up data!')
        with open(f'rw_demo_scores_1206{N//25}.json', 'w') as f:
            json.dump(run_data, f, indent=True)

with open(f'rw_demo_scores_all_1206.json', 'w') as f:
        json.dump(run_data, f, indent=True)
