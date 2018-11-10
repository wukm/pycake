#!/usr/bin/env python3

from skimage.morphology import disk, binary_erosion, binary_dilation
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries

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

#############
# DEMO FOR SHOWING OFF DILATE_BOUNDARY EFFECT

if __name__ == "__main__":

    from placenta import get_named_placenta
    from frangi import frangi_from_image
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.exposure import rescale_intensity

    import os.path

    dest_dir = 'demo_output'
    img = get_named_placenta('T-BN0164923.png')

    #radius = 30
    sigma = 3
    radius = 80
    # IMG WITHOUT DILATING, THEN IMAGE WITH DILATING
    #sigma = radius/4

    #inset = np.s_[:,:]
    inset = np.s_[800:1000,500:890]
    #inset = np.s_[100:300,300:500]

    D = dilate_boundary(img, radius=radius)


    # SAVE IT IN THE RIGHT DIRECTORY, ETC plt.savefig(

    # NOW SHOW FRANGI ON THESE IMAGES

    # NOW SHOW THE SAME PICTURE

    Fimg = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=None)
    #FD = frangi_from_image(img, sigma, dark_bg=False, dilation_radius=radius)
    FD = frangi_from_image(D, sigma, dark_bg=False)

    #Fimg = rescale_intensity(Fimg)
    #FD = rescale_intensity(FD)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 7))

    axes[0,0].imshow(img[inset].filled(0), cmap=plt.cm.gray)
    axes[0,1].imshow(D[inset].filled(0), cmap=plt.cm.gray)

    axes[1,0].imshow(Fimg[inset].filled(0), cmap=plt.cm.nipy_spectral)
    axes[1,1].imshow(FD[inset].filled(0), cmap=plt.cm.nipy_spectral)

    # this is a hack directly from matplotlib, that's why it's so ugly.
    plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axes[:, 1]], visible=False)

    fig.tight_layout()

    plt.savefig(os.path.join(dest_dir, "boundary_dilation_demo.png"), dpi=300)

