#!/usr/bin/env python3

"""
currently obsolete. this code has been merged into preprocessing.
this file should ideally produce a visual montage (or individual files)
showing the success/failure of find_plate_in_raw
"""
from placenta import (get_named_placenta, list_placentas, open_typefile,
                      show_mask)
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import mask_stump
import numpy.ma as ma
import sys
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_int
from hfft import fft_gradient

from skimage.segmentation import watershed
import scipy.ndimage as ndi
from skimage.morphology import binary_erosion, disk

def find_plate_in_raw(raw, sigma=.01):
    g = fft_gradient(raw[...,1],sigma=.01)
    marks=np.zeros(img.shape, np.int32)
    marks[0,0] = 1
    marks[g > g.mean()] = 2
    #marks[g > np.percentile(g,25)] = 2
    w = watershed(g,marks)

    eroded = binary_erosion(w==2, disk(15))

    labeled, n_labs = ndi.label(eroded)

    # get largest object (0 is gonna be background)
    # sort labels by decreasing magnitude
    labs_by_size = sorted(list(range(1,n_labs+1)),
                          key=lambda l: np.sum(labeled==l), reverse=True)

    # unless something went horribly wrong
    plate_index = labs_by_size[0]

    return ~(labeled == plate_index)

FAILS = [
    "T-BN0687730.png",
    "T-BN1629357.png",
    "T-BN2050224.png",
    "T-BN6381701.png",
    "T-BN7476220.png",
    "T-BN7644170.png",
    "T-BN7767693.png",
]

for n, filename in enumerate(FAILS):

    print(filename)

    raw = open_typefile(filename, 'raw')
    img = get_named_placenta(filename)

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(30,12))

    ax[0].imshow(raw)
    ax[0].set_title(filename)

    plate = find_plate_in_raw(raw)
    newmask = show_mask(ma.masked_array(img, mask=plate))
    ax[1].imshow(newmask)
    ax[2].imshow(plate*1. + img.mask*2, vmin=0, vmax=3)

    plt.show()

    if input('make more?') == 'n':
        sys.exit(0)
