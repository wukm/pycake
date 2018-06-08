#!/usr/bin/env python3

"""
Get registered, unpreprocessed placental images.  No automatic registration
(i.e. segmentation of placental plate) takes place here. The background,
however, *is* masked.

Again, there is no support for unregistered placental pictures.
Any region outside of the placental plate MUST be black.

There is currently no support for color images.

TODO:
    - Build sample base & organize data :v)
    - Test on many other images.
    - Think of how the interface should really work, esp for get_named_placenta
    - Fix logic in mask_background
    - Catch errors better.
    - Support for color images
    - Show a better test
    - Be able to grab trace files too.
    - Cache masked samples.
"""

import numpy as np
import numpy.ma as ma
from skimage import segmentation, morphology
import os.path

from scipy.ndimage import imread

def get_named_placenta(filename, sample_dir=None, masked=True, mask=None):
    """
    This function is to be replaced by a more ingenious/natural
    way of accessing a database of unregistered and/or registered
    placental samples.

    INPUT:
        filename: name of file (including suffix?) but NOT directory
        masked: return it masked.
        mask: if supplied, this will use a supplied one channel mask. otherwise,
            this will be calculated automatically.
        sample_directory: None. defaults to './samples'

    """
    if sample_dir is None:
        sample_dir = 'samples'

    full_filename = os.path.join(sample_dir, filename)

    raw_img = imread(full_filename, mode='L')

    if mask is None:
        return mask_background(raw_img)
    else:
        mask = os.path.join(sample_dir, mask)
        mask = imread(mask, mode='L')
        return ma.masked_array(raw_img, mask=np.invert(mask))

def mask_background(img):
    """
    Masks all regions of the image outside the placental plate.
    The logic of the function could be improved.

    INPUT:
        img:
            A color or grayscale array corresponding to an image of a placenta
            with the plate in the 'middle.' Outer regions should be black.

    OUTPUT:
        masked_img:
            A numpy.ma.masked_array with the same dimensions.
    """

    if img.ndim == 3:

        #mark any pixel with with content in any channel
        bg_mask = img.any(axis=-1)
        bg_mask = np.invert(bg_mask)

        # make the mask multichannel to match dim of input
        bg_mask = np.repeat(bg_mask[:,:,np.newaxis], 3, axis=2)

    else:

        # same as above
        bg_mask = (img != 0)
        bg_mask = np.invert(bg_mask)

    # the above approach will probably work for any real image (i.e. a
    # photgraph). it will obviously fail for any image where there is true black
    # in the placental plane. This should work instead:

    # find the outer boundary and mark outside of it.
    # run with defaults, sufficient
    bound = convex_hull_image(bg_mask)
    bound = segmentation.find_boundaries(bg_mask, mode='inner', background=1)
    bg_mask[bound] = 1

    #remove any small holes found inside the plate (regions or single pixels
    #that happen to be black).  run with defaults, sufficient
    holes = morphology.remove_small_holes(bg_mask)
    bg_mask[holes] = 1

    return ma.masked_array(img, mask=bg_mask)


if __name__ == "__main__":
    """test that this works on an easy image."""

    from scipy.ndimage import imread
    import matplotlib.pyplot as plt

    filename = 'raw_barium.png'

    img =  get_named_placenta('raw_barium.png')
