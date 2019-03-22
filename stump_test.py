#!/usr/bin/env python3

"""
demonstrate the success failure of an automated way to remove the
umbilical stump from each image. the method, which can be found in
preprocessing.mask_stump, works by threshold on the image (170/255) and then
performing a white top hat.

another method,
"""

from placenta import (get_named_placenta, list_placentas, open_typefile,
                      show_mask, measure_ncs_markings)
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import mask_stump
import numpy.ma as ma
import sys
from skimage.exposure import equalize_adapthist
from scipy.ndimage import label
from skimage.morphology import thin, binary_opening, disk, convex_hull_image
import seaborn as sns


#sns.set(font_scale=0.8)

for n, filename in enumerate(list_placentas('T-BN')):

    print(filename)

    raw = open_typefile(filename, 'raw')
    #raw = (equalize_adapthist(raw))
    img = get_named_placenta(filename)

    # get a binary mask of where large high brightness blobs are
    stump = mask_stump(raw, mask=img.mask, mask_only=True)
    if not stump.any():
        print('no stump found at all, continuing')
        plt.imshow(raw)
        plt.show()
        plt.close()
        continue

    stump_labs = label(stump)

    ucip_mid, _ = measure_ncs_markings(filename=filename)

    labeled, n_labs = label(stump)
    dists = list()
    for lab in range(1, n_labs+1):
        X, Y = np.where(labeled==lab)
        # make a tuple of stump label number and minimum distance between
        # the ucip midpoint of pixels with that label
        dists.append((lab, min([(x-ucip_mid[0])**2 + (y-ucip_mid[1])**2
                        for x,y in zip(X,Y)])))

    # find the closest label and its distance
    closest_lab, closest_dist = sorted(dists, key=lambda v: v[1])[0]

    # get a mask of the stump (just one label)
    closest_stump = (labeled==closest_lab)

    # perform a binary opening of it
    closest_opening = binary_opening(closest_stump, disk(10))
    # and take its convex hull

    final_stump = convex_hull_image(closest_opening)

    fig, ax = plt.subplots(ncols=3, nrows=2)

    ax[0,0].imshow(raw)
    ax[0,0].set_title(filename)

    newmask = show_mask(ma.masked_array(img, mask=stump))
    ax[0,1].imshow(newmask)
    ax[0,1].set_title('masked image after mask_stump')
    ax[0,2].imshow(stump*1. + img.mask*2, vmin=0, vmax=3)
    ax[0,2].set_title('potential stumps found by mask_stump'
                      '(with original mask)')
    ax[1,0].imshow(closest_stump)
    ax[1,0].set_title('closest stump to reported UCIP midpoint')
    ax[1,1].imshow(closest_opening)
    ax[1,1].set_title('binary opening of closest stump (radius 10)')
    ax[1,2].imshow(final_stump)
    ax[1,2].set_title('convex hull of binary opening')

    [a.axis('off') for a in ax.ravel()]


    plt.show()
    plt.close()

    if input('make more?') == 'n':
        sys.exit(0)
