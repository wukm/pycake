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

def get_named_placenta(filename, sample_dir=None, masked=True, maskfile=None):
    """
    This function is to be replaced by a more ingenious/natural
    way of accessing a database of unregistered and/or registered
    placental samples.

    INPUT:
        filename: name of file (including suffix?) but NOT directory
        masked: return it masked.
        maskfile: if supplied, this use the file will use a supplied 1-channel
                mask (where 1 represents an invalid/masked pixel, and 0
                represents a valid/unmasked pixel. the supplied image must be
                the same shape as the image. if not provided, the mask is
                calculated (unless masked=False)
                the file must be located within the sample directory
        sample_directory: Relative path where sample (and mask file) is located.
                defaults to './samples'

    if masked is true (default), this returns a masked array.

    NOTE: A previous logical incongruity has been corrected. Masks should have
    1 as the invalid/background/mask value (to mask), and 0 as the
    valid/plate/foreground value (to not mask)
    """
    if sample_dir is None:
        sample_dir = 'samples'

    full_filename = os.path.join(sample_dir, filename)

    raw_img = imread(full_filename, mode='L')

    if maskfile is None:
        print("PLEASE SUPPLY A MASKFILE. AUTOGENERATION OF MASKS IS SLOW / BUGGED.")
        return mask_background(raw_img)
    else:
        # set maskfile name relative to path
        maskfile = os.path.join(sample_dir, maskfile)
        mask = imread(maskfile, mode='L')
        return ma.masked_array(raw_img, mask=mask)

def mask_background(img):
    """
    THIS IS BROKEN/ VERY SLOW.  PROVIDE MANUAL MASKS OR FIX.
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
    bound = morphology.convex_hull_image(bg_mask)
    bound = segmentation.find_boundaries(bg_mask, mode='inner', background=1)
    bg_mask[bound] = 1

    #remove any small holes found inside the plate (regions or single pixels
    #that happen to be black).  run with defaults, sufficient
    holes = morphology.remove_small_holes(bg_mask)
    bg_mask[holes] = 1

    return ma.masked_array(img, mask=bg_mask)

def mimshow(img):
    """
    show a masked grayscale image with a dark blue masked region

    custom version of imshow that shows grayscale images with the right colormap
    and, if they're masked arrays, sets makes the mask a dark blue)
    a better function might make the grayscale value dark blue
    (so there's no confusion)
    """

    from numpy.ma import is_masked
    from skimage.color import gray2rgb
    import matplotlib.pyplot as plt


    if not is_masked(img):
        plt.imshow(img, cmap=plt.cm.gray)
    else:

        mimg = gray2rgb(img.filled(0))
        # fill blue channel with a relatively dark value for masked elements
        mimg[img.mask, 2] = 60
        plt.imshow(mimg)

if __name__ == "__main__":

    """test that this works on an easy image."""

    from scipy.ndimage import imread
    import matplotlib.pyplot as plt
    test_filename = 'barium1.png'
    test_maskfile = 'barium1.mask.png'

    img =  get_named_placenta(test_filename, maskfile=test_maskfile)

    print('run plt.show() to see masked output')
    mimshow(img)

