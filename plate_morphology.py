#!/usr/bin/env python3

from skimage.morphology import disk, binary_erosion, binary_dilation
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries

import numpy as np
import numpy.ma as ma

def erode_plate(img, erosion_radius=20, plate_mask=None):
    """
    Manually remove (erode) the outside boundary of a plate.
    The goal is remove any influence of the zeroed background
    on reporting derivative information

    NOTE: this is an old deprecated function. use dilate boundary instead.
    """
    if plate_mask is None:
        # grab the mask from input image
        try:
            plate_mask = img.mask
        except AttributeError:
            raise('Need to supply mask information')

    # convex_hull_image finds white pixels
    plate_mask = np.invert(plate_mask)

    # find convex hull of mask (make erosion calculation easier)
    plate_mask = np.invert(convex_hull_image(plate_mask))

    # this is much faster than a disk. a plus sign might be better even.
    #selem = square(erosion_radius)
    # also this correctly has erosion radius as the RADIUS!
    # input for square() and disk() is the diameter!
    selem = np.zeros((erosion_radius*2 + 1, erosion_radius*2 + 1),
                        dtype='bool')
    selem[erosion_radius,:] = 1
    selem[:,erosion_radius] = 1
    eroded_mask = binary_erosion(plate_mask, selem=selem)

    # this is by default additive with whatever
    return ma.masked_array(img, mask=eroded_mask)

def dilate_boundary(img, radius=10, mask=None):
    """
    grows the mask by a specified radius of a masked 2D array
    Manually remove (erode) the outside boundary of a plate.
    The goal is remove any influence of the zeroed background
    on reporting derivative information.

    There is varying functionality here (maybe should be multiple functions
    instead?)

    If img is a masked array and mask=None, the mask will be dilated and a
    masked array is outputted.

    If img is any 2D array (masked or unmasked), if mask is specified, then
    the mask will be dilated and the original image will be returned as a
    masked array with a new mask.

    If the img is None, then the specified mask will be dilated and returned
    as a regular 2D array.

    """

    if mask is None:
        # grab the mask from input image
        # if img is None this will break too but not handled
        try:
            mask = img.mask
        except AttributeError:
            raise('Need to supply mask information')

    perimeter = find_boundaries(mask, mode='inner')

    maskpad = np.zeros_like(perimeter)

    M,N = maskpad.shape
    for i,j in np.argwhere(perimeter):
        # just make a cross shape on each of those points
        # these will silently fail if slice is OOB thus ranges are limited.
        maskpad[max(i-radius,0):min(i+radius,M),j] = 1
        maskpad[i,max(j-radius,0):min(j+radius,N)] = 1

    new_mask = np.bitwise_or(maskpad, mask)

    if img is None:
        return new_mask # return a 2D array
    else:
        # replace the original mask or create a new masked array
        return ma.masked_array(img, mask=new_mask)
