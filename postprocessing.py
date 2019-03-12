#!/usr/bin/env python3

"""
Contains some of the nontrivial segmentation methods.
"""

from skimage.filters import sobel
from skimage.morphology import (remove_small_objects, thin, disk,
                                binary_dilation)

from frangi import frangi_from_image
from merging import apply_threshold, nz_percentile
from plate_morphology import dilate_boundary
from skimage.segmentation import random_walker
import numpy as np

from skimage.filters.rank import enhance_contrast_percentile as ecp
from scipy.ndimage import distance_transform_edt as edt


def random_walk_fill(img, Fmax, high_thresh, low_thresh, dark_bg):
    """
    # this is deprecated, it's trash
    """

    s = sobel(img)
    s = dilate_boundary(s, mask=img.mask, radius=20)

    finv = frangi_from_image(img, sigma=0.8, beta=0.5, dark_bg=(not dark_bg),
                             dilation_radius=20)

    finv_thresh = (finv > nz_percentile(finv, 50)).filled(0)
    margins = remove_small_objects(finv_thresh, min_size=32)

    markers = np.zeros(img.shape, dtype=np.int32)
    markers[Fmax < low_thresh] = 1

    margins_added = (margins | (Fmax > high_thresh))
    #margins_added = remove_small_holes(margins_added, area_threshold=50)

    markers[Fmax < low_thresh] = 1

    markers[margins_added] = 2

    rw = random_walker(1-Fmax, markers, beta=1000)

    approx_rw = (rw == 2)

    return approx_rw, markers, margins_added


def random_walk_scalewise(F, high_thresh=0.4, rw_beta=130,
                          return_labels=False):
    """Random walker on each a multiscale Frangi result"""
    print('doing scalewise random walk', end=' ')
    V = np.transpose(F, axes=(2, 0, 1))
    W = np.zeros(V.shape, np.bool)
    for n, v in enumerate(V):
        print('σ', end='', flush=True)
        markers = np.zeros(v.shape, np.int32)
        markers[v == 0] = 1
        # this could be a vector too
        markers[v > high_thresh] = 2
        # or 1-v
        W[n] = (random_walker(v, markers, rw_beta) == 2)
    print()
    if not return_labels:
        return W.any(axis=0)
    else:
        # argmax grabs the first scale where it was satisfied
        # so this will grab the lowest scale that matches
        return W.any(axis=0), W.argmax(axis=0)


def dilate_to_rim(spine, rim, thin_spine=False, max_radius=15,
               return_radii=False):
    """Grow spine region by dilating towards closest rim point of the trough.

    Parameters:
    -----------
    spine, rim: 2D ndarray type bool
        spine are points, dilate to rim
    thin_spine: bool, optional
        Thin the spine before performing dilation. This makes it much faster
        at the risk of giving an incomplete fill if the spine isn't in the
        middle of the margins, also makes returning the labels more meaningful

    Returns
    -------
    ...

    """

    D = np.ones(spine.shape, np.bool)
    D[rim] = 0

    spine_dists = edt(D)

    # either thin first or don't, i dunno
    if thin_spine:
        spine_dists[~thin(spine)] = 0
    else:
        spine_dists[~spine] = 0

    spine_radii = np.round(spine_dists).astype('int')

    dilation_stack = np.stack([binary_dilation(spine_radii==r, selem=disk(r))
                               for r in range(1,max_radius+1)])

    approx = dilation_stack.any(axis=0)

    if not return_radii:
        return approx
    else:
        return approx, dilation_stack.argmax(axis=0)
