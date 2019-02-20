#!/usr/bin/env python3

"""
this shows a step by step of the "dilate_to_margin" or "trough_filling"
algorithm based on morphology and the frangi filter
"""

import numpy as np
import matplotlib.pyplot as plt
from placenta import (list_by_quality, cropped_args, mimg_as_float,
                      open_typefile, open_tracefile, strip_ncs_name)

# rename this elsewhere
from placenta import get_named_placenta as prepare_placenta

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile, apply_threshold
from plate_morphology import dilate_boundary
import os.path, os

from scipy.ndimage import distance_transform_edt as edt
from skimage.filters.rank import enhance_contrast_percentile as ecp
from skimage.morphology import disk, binary_dilation, thin
from scoring import confusion, mcc

from skimage.filters import threshold_isodata
from skimage.exposure import rescale_intensity
from postprocessing import dilate_to_rim
from preprocessing import inpaint_hybrid
import json


def split_signed_frangi_stack(F, negative_range=None, positive_range=None,
                              mask=None):
    """
    F is the frangi stack where the first dimension is the scale space.
    transposing it would be easier.

    if negative_range is given, then only scales within that range are
    accumulated

    will return Vmax(+) and Vmin(-)
    """

    if negative_range is None:
        negative_range = (0, F.shape[-1])

    if positive_range is None:
        positive_range = (0, F.shape[-1])

    f = F[positive_range[0]:positive_range[1]].max(axis=0)
    nf = ((-F*(F<0))[negative_range[0]:negative_range[1]]).max(axis=0)

    if mask is not None:
        nf[mask] = 0
        f[mask] = 0

    return f, nf


# probably just make this a list of two so you can show
placentas = list_by_quality(0, N=1)

OUTPUT_DIR = 'output/190219_trough_demo'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# overall strict Frangi parameters
beta =0.15
gamma = 1.0
log_range = (-1.5, 3.2)
N_scales = 20
THRESHOLD = .3
MARGIN_THRESHOLD = 0.01
NEGATIVE_RANGE = (0,6)

scales = np.logspace(*log_range, base=2, num=N_scales)

# these are used to store our scores for each sample
mccs = list()
precs = list()

for filename in placentas:

    # get the name of the sample (like 'BN#######')
    basename = strip_ncs_name(filename)

    print(basename, '*'*30)

    # this calls find_plate_in_raw for all files not in placenta.FAILS
    # in an attempt to improve upon nthe border
    # load sample and do pre-processing
    img = prepare_placenta(filename)
    img = inpaint_hybrid(img)

    # load trace
    crop = cropped_args(img)
    trace = open_tracefile(filename)
    bigmask = dilate_boundary(None, mask=img.mask, radius=20)  # check that this works

    # make a MxNxN_scales matrix of Frangi outputs (where (M,N) == img.shape)
    F = np.stack([frangi_from_image(img, sigma, beta, gamma, dark_bg=False,
                                    dilation_radius=20, rescale_frangi=True,
                                    signed_frangi=True).filled(0)
                                    for sigma in scales])

    f, nf = split_signed_frangi_stack(F, positive_range=None,
                                      negative_range=NEGATIVE_RANGE,
                                      mask=bigmask)


    # these next few lines just process a visual phenomenon
    # just in case Vmax(+) and Vmax(-) are both nonzero at the same pixel, we
    # will prioritize Vmax if it is above THRESHOLD, otherwise Vmin if it is
    # above MARGIN threshold, otherwise, Vmax(+)
    # not sure what i really want to show here, maybe just Vmax(+) or just a
    # each over thresholds (i.e. a 3 color map)

    recolor_with_negative = (f <= THRESHOLD) & (nf > MARGIN_THRESHOLD)

    fboth = f.copy()
    fboth[recolor_with_negative] = -1

    spine = dilate_boundary(f, mask=img.mask, radius=20).filled(0)
    spine = rescale_intensity(spine, in_range=(0,1), out_range='uint8')
    ecp_spine = spine.astype('uint8')
    ecp_spine = ecp(ecp_spine, disk(3)) > (THRESHOLD*255)

    fixed = f > THRESHOLD
    margins = (nf > MARGIN_THRESHOLD)
    # just make two different colored markers, this is hacky
    markers = np.zeros_like(spine)
    markers[fixed] = 0.5
    markers[margins] = -0.5

    # trough filling with ECP prefilter
    #approx_td_ecp, radii = dilate_to_rim(ecp_spine, margins, thin_spine=False, return_radii=True)
    approx, radii = dilate_to_rim(fixed, margins, thin_spine=True, return_radii=True)
    approx2, radii2 = dilate_to_rim(fixed, margins, return_radii=True)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25,15), sharex=True,
                           sharey=True)
    ax[0,0].imshow(img[crop], cmap=plt.cm.gray)
    ax[0,1].imshow(f[crop], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[0,2].imshow(-nf[crop], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[1,0].imshow(markers[crop], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[1,1].imshow(confusion(approx2,trace)[crop])
    ax[1,2].imshow(confusion(approx2 & ~fixed,trace)[crop])

    [a.axis('off') for a in ax.ravel()]
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(os.path.join(OUTPUT_DIR, ''.join(('fig-', basename, '.png'))))
    plt.show()

    plt.close()
    inset = np.s_[151:486,147:653]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25,15), sharex=True,
                           sharey=True)
    ax[0,0].imshow(img[crop][inset], cmap=plt.cm.gray)
    ax[0,1].imshow(f[crop][inset], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[0,2].imshow(-nf[crop][inset], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[1,0].imshow(markers[crop][inset], vmax=1.0, vmin=-1.0, cmap=plt.cm.seismic_r)
    ax[1,1].imshow(confusion(approx2,trace)[crop][inset])
    ax[1,2].imshow(confusion(approx2 & ~fixed,trace)[crop][inset])

    [a.axis('off') for a in ax.ravel()]
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(os.path.join(OUTPUT_DIR, ''.join(('fig-inset', basename, '.png'))))
    plt.show()
