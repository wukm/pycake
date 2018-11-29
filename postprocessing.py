#!/usr/bin/env python3

"""
doing things to the Frangi targets, i.e. feeding them into other algorithms
"""

from skimage.filters import sobel
from skimage.morphology import remove_small_holes, remove_small_objects, thin
from frangi import frangi_from_image
from merging import apply_threshold, nz_percentile
from plate_morphology import dilate_boundary
from skimage.segmentation import random_walker
import numpy as np

def random_walk_fill(img, Fmax, high_thresh, low_thresh, dark_bg):
    """
    there are a lot of decisions to make here but it's just a demo
    so who cares
    """

    s = sobel(img)
    s = dilate_boundary(s, mask=img.mask, radius=20)

    finv = frangi_from_image(img, sigma=0.3, beta=0.35, dark_bg=(not dark_bg),
                             dilation_radius=20)

    finv_thresh = (finv > nz_percentile(finv, 80)).filled(0)
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
