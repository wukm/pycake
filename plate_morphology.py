#!/usr/bin/env python3

from skimage.morphology import disk, binary_erosion, binary_dilation
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries, watershed

from placenta import open_typefile, get_named_placenta

import numpy as np
import numpy.ma as ma

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


def mask_cuts(img, ucip, mask_only=False, in_place=False, return_success=False):
    """

    this doesn't handle any image, io. just provide the ucip img and the
    base (masked) image and we'll fix the mask

    ucip is the actual RGB array, not the file. do io elsewhere.

    if mask_only, this will simply return the new mask as a 2D boolean array.
    Otherwise, it returns a masked_array.
    The cut region will be added to the img's mask. If you really want just the
    difference, you'll have to to run
    >>>np.logical_and(cut_mask, np.invert(img.mask)) youself.

    yourself.

    If in_place, this changes the mask of the image directly (but still returns
    a masked array. If mask_only is True, in_place will automatically be set to
    False to prevent hideous side effects

    if return_success, this function returns True if there was a cutmark found,
    otherwise false as a second output
    """
    # get indices where the blue square indicating center of a cut apprears
    cutmarks = np.all(ucip==(0,0,255), axis=-1)

    if not np.any(cutmarks):

        #print("no cutmarks found on image")

        if return_success:
            return img, False
        else:
            return img
    else:
        #print("found a cutmark!")
        pass

    cutmarks = np.nonzero(cutmarks)
    # get the first pixel of it (we don't need to be too precise here)
    X, Y = cutmarks[0][0], cutmarks[1][0]

    # get a value somewhat lower than the value of bg in the cut
    # (this should be a high number before we take 85%)
    # sometimes this is in a shadowy region which fucks everything up though
    threshold = max(img[cutmarks].mean() * .85, 175)

    rmin, rmax = max(0, X-100), min(img.shape[0], X+100)
    cmin, cmax = max(0, Y-100), min(img.shape[1], Y+100)
    cutregion = np.s_[rmin:rmax, cmin:cmax] # get a window around the mark

    # mark inside of the placenta with label 2, original mask and cutmarks with
    # label 1, and the rest with 0 (i dunno)
    markers = np.zeros(img.shape, dtype='int32')
    markers[img.filled(255) < threshold] = 2
    markers[img.mask] = 1
    markers[cutmarks] = 1

    # perform watershedding on the thresholded image to fill in the cut with
    # label 1
    cutfix = watershed(img.filled(255) < threshold, markers=markers)

    # this is a waste considering the in_place, but eh
    new_mask = img.mask.copy()

    new_mask[cutregion] = (cutfix[cutregion] == 1)

    if mask_only:

        out = new_mask

    elif not in_place:

        out = ma.masked_array(img, mask=new_mask)

    else:

        # will this work?
        img[new_mask] = ma.masked

        out = img

    # now return succeed if asked to
    if return_success:
        return out, True

    else:
        return out


if __name__ == "__main__":

    # DEMO FOR SHOWING OFF DILATE_BOUNDARY EFFECT

    from placenta import get_named_placenta
    from frangi import frangi_from_image
    import matplotlib.pyplot as plt

    import os.path

    dest_dir = 'demo_output'
    img = get_named_placenta('T-BN0164923.png')

    sigma = 3
    radius = 25

    inset = np.s_[800:1000, 500:890]

    D = dilate_boundary(img, radius=radius)

    Fimg = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=None)
    FD = frangi_from_image(D, sigma, dark_bg=False)
    FDinv = frangi_from_image(D, sigma, dark_bg=True)
    Finv = frangi_from_image(img, sigma, dark_bg=True, dilation_radius=None)

    fig, axes = plt.subplots(ncols=2, nrows=3)

    axes[0,0].imshow(img[inset].filled(0), cmap=plt.cm.gray)
    axes[0,1].imshow(D[inset].filled(0), cmap=plt.cm.gray)
    axes[1,0].imshow(Fimg[inset].filled(0), cmap=plt.cm.nipy_spectral)
    axes[1,1].imshow(FD[inset].filled(0), cmap=plt.cm.nipy_spectral)
    axes[2,0].imshow(Finv[inset].filled(0), cmap=plt.cm.nipy_spectral)
    axes[2,1].imshow(FDinv[inset].filled(0), cmap=plt.cm.nipy_spectral)

    for a in axes.ravel():
        # get rid of all the labels
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)

    # lol matlab
    for i in range(5):
        fig.tight_layout()

    plt.savefig(os.path.join(dest_dir, "boundary_dilation_demo.png"), dpi=300)

