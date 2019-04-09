#!/usr/bin/env python3

"""
for a single sample
skeletonize the ground truth widths
calculate the Frangi filter response at these pixels.
For each binned width in ground truth,
    make a histogram of Vargmax
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile,
                      strip_ncs_name, list_placentas, measure_ncs_markings,
                      add_ucip_to_mask)

from frangi import frangi_from_image
from merging import nz_percentile, apply_threshold
from plate_morphology import dilate_boundary
import os.path
import os

# from scipy.ndimage import distance_transform_edt as edt
# from skimage.filters.rank import enhance_contrast_percentile as ecp
from scoring import confusion, mcc

from skimage.color import grey2rgb

from skimage.filters import threshold_isodata
from postprocessing import dilate_to_rim
from preprocessing import inpaint_hybrid
import datetime
import json

import numpy.ma as ma

from scoring import merge_widths_from_traces
from skimage.morphology import thin

from scipy.ndimage import maximum_filter
from skimage.morphology import disk


# or use a particular sample name
filename = list_by_quality(0)[1]
basename = strip_ncs_name(filename) # get the name of the sample ( 'BN#######')
print(basename, '*'*30)

# Uncomment one of the parametrizatons
#beta, gamma, parametrization_name = 0.15, 1.0, "strict"
beta, gamma, parametrization_name = 0.35, 0.5, "semistrict"
#beta, gamma, parametrization_name = 0.5, 1.0, 'semistrict-gamma'
#beta, gamma, parametrization_name = 0.5, 0.5, "standard"

today = f"{datetime.datetime.now():%y%m%d}"
OUTPUT_DIR = (f'demo_output/{today}-width_to_scale_response_demo-unscaled')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Other basic setup
scales = np.arange(0.2, 18.2, step=0.2)
min_response = .05 # unused currently


# this calls find_plate_in_raw for all files not in placenta.FAILS
# in an attempt to improve upon the border
img = get_named_placenta(filename)
img = inpaint_hybrid(img)
ucip, res = measure_ncs_markings(filename=filename)
old_mask = img.mask.copy()
img.mask = add_ucip_to_mask(ucip, radius=50, mask=img.mask)
A_trace = open_typefile(filename, 'arteries')
V_trace = open_typefile(filename, 'veins')

# load trace
crop = cropped_args(img)
cimg = open_typefile(filename, 'raw')

# not for actual analysis, overlap is merged. better to use A_trace and V_trace
ctrace = open_typefile(filename, 'ctrace')
trace = open_tracefile(filename)
bigmask = dilate_boundary(None, mask=img.mask, radius=20)

# rotate the image so it's hotdog not hamburger
if img[crop].shape[0] > img[crop].shape[1]:
    # and rotating it would be fix all this automatically
    cimg = np.rot90(cimg)
    ctrace = np.rot90(ctrace)
    trace = np.rot90(trace)
    img = np.rot90(img)
    crop = cropped_args(img)
    bigmask = dilate_boundary(None, mask=img.mask, radius=20)
    old_mask = np.rot90(old_mask)
    A_trace = np.rot90(A_trace)
    V_trace = np.rot90(V_trace)

# 2D matrix of reported vessel widths 3,5,7,...,19
# widths.astype('bool') == trace should be true.
widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')

# skeletonize trace (do separately, so overlaps aren't weird)
A_skel = thin(~np.all(A_trace == (255,255,255), axis=-1))
V_skel = thin(~np.all(V_trace == (255,255,255), axis=-1))
skel = (A_skel | V_skel)

# keep original widths, but we only really care about the ones on the ridge
skelwidths = widths.copy()
skelwidths[~skel] = 0

V = list() # store each Frangi filter scale

for n, sigma in enumerate(scales,1):
    # want to calculate hessian at all points to get accurate S_max
    # but then you can throw things away
    print(f'{n}:\t{sigma:.02}', end='\t', flush=True)
    v = frangi_from_image(img, sigma, beta, gamma, dark_bg=False,
                          dilation_radius=20, rescale_frangi=False)
    fname = os.path.join(OUTPUT_DIR, f'{basename}-Vsigma-{n:02}-unscaled.png')
    plt.imsave(fname, v[crop].filled(0),
               cmap=plt.cm.nipy_spectral, vmin=0, vmax=1)
    # throw out things that aren't part of the ground truth
    # skel is too much for now
    v[~trace] = 0
    v[v < .10] = 0
    V.append(v.filled(0))
print()
V = np.stack(V) # only has values along ground truth shape (n_scales, *img.shape)

#Vmax = V.max(axis=0)
# scale (index) at which maximal Frangi value occurs
# masking out things not on
#Vargmax = ma.masked_array(V.argmax(axis=0), mask=~skel)

sns.set() # set seaborn aesthetic defaults

for w in range(3,20,2):
    #scale_responses = Vargmax[skelwidths==w]
    # make the disk one smaller to try to not include any margin points
    #local_radius = max((w - 1)//2 - 1, 1) # make it at least one though
    local_radius = (w - 1) // 2
    local_foot = disk(local_radius)
    # replace the filter response on ridge with the maximal response in the
    # nearby vicinity
    print(f'width {w}')
    V_local = np.stack([maximum_filter(v,footprint=local_foot) for v in V])

    # now find index scale contains the maximal response (masking things not on
    # skel)
    V_argmax = ma.masked_array(V_local.argmax(axis=0), mask=~skel)
    #plt.imshow(V_argmax[crop])
    fname = os.path.join(OUTPUT_DIR, f'{basename}-Vlocal_max-{w:02}-unscaled.png')
    plt.imsave(fname, (skel*V_local.max(axis=0))[crop],
               cmap=plt.cm.nipy_spectral, vmin=0, vmax=1)
    scale_responses = V_argmax[skelwidths==w]
    fig = plt.figure(figsize=(6.5, 3))
    plt.hist(scale_responses.filled(100), bins=range(len(scales)+1), density=False)
    plt.title(r'$\mathcal{V}_{\arg\max}$'
              f' for skeltrace pixels of binned width {w}')
    # try do so some sort of sensible labeling, using the actual sigma size
    s = np.hstack(([0,], scales))
    plt.xticks(np.arange(0,95,10), [f'{x:.1f}' for x in s[::10]])

    plt.xlabel(r'scale of maximal response ($\sigma$)')
    #plt.ylabel('# pixels')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'{basename}-scale_to_argmax_hist-{w:02}-unscaled.png')
    plt.savefig(fname)
