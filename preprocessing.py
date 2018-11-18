#!/usr/bin/env python3

from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.restoration import inpaint_biharmonic
import numpy as np
import numpy.ma as ma
from scipy.ndimage import label
from skimage.util import img_as_float
from skimage.segmentation import find_boundaries
from plate_morphology import dilate_boundary


def inpaint_glare(img, threshold=175, window_size=15, mask=None):
    """
    img is a masked array type uint [0,255]
    """

    # bool array, true where glare
    if mask is None:
        glared = mask_glare(img, threshold=threshold, mask_only=True)
    else:
        glared = mask

    B = ma.masked_array(img, mask=glared)  # masked background *and* glare
    new_img = img.copy()  # copy values of original image (will rewrite)
    d = int(window_size)

    for j, k in zip(*np.where(glared)):
        # rewrite all glared pixels with the mean of nonmasked elements
        # in a window_size window. (this doesn't check OoB, be careful!)
        new_img[j, k] = B[j-d:j+d, k-d:k+d].compressed().mean()

    return new_img


def inpaint_with_boundary_median(img, threshold=175, mask=None):
    """
    mask glare pixels, then replace by the median value on the mask's boundary
    """
    if mask is None:
        glared = mask_glare(img, threshold=threshold, mask_only=True)
    else:
        glared = mask

    B = ma.masked_array(img, mask=glared)

    new_img = img.copy()  # copy values of original image (will rewrite)
    bounds = find_boundaries(glared)
    lb, _ = label(bounds)
    fill_vals = np.zeros_like(img.data)

    # for each boundary of masked region, find the median value of the img
    for lab in range(1, lb.max()+1):
        inds = np.where(lb == lab)
        fill_vals[inds] = nz_median(B[inds])

    # label masked regions together with their boundaries (they'll be
    # connected)
    lm, _ = label(np.logical_or(glared, lb != 0))

    # fill the masked areas with the corresponding fill value
    for lab in range(1, lm.max()+1):
        inds = np.where(lm == lab)
        # find locations of filled values corresponding to this label
        # median in case there's overlapped regions? (sloppy)
        replace_value = nz_median(fill_vals[inds])

        if replace_value == 0:
            raise

        fill_vals[inds] = replace_value

    # now fill in the values
    new_img[glared] = fill_vals[glared]

    return new_img


def nz_median(A):

    if ma.is_masked(A):
        relevant = A[A > 0].compressed()
    else:
        relevant = A[A > 0]

    return np.median(relevant)


def inpaint_hybrid(img, threshold=175, min_size=64, boundary_radius=10):
    """
    use biharmonic inpainting in larger, inner areas (important stuff)
    and median inpainting in smaller areas and along boundary
    """

    glare = mask_glare(img, threshold=threshold, mask_only=True)

    glare_inside = dilate_boundary(glare, mask=img.mask,
                                   radius=boundary_radius).filled(0)

    large_glare = remove_small_objects(glare_inside, min_size=min_size,
                                       connectivity=2)
    small_glare = np.logical_and(glare, np.invert(large_glare))

    # inpaint smaller and less important values with less expensive method
    inpainted = inpaint_with_boundary_median(img, mask=small_glare)
    hybrid = img_as_float(inpainted) # scale 0 to 1

    # inpaint larger regions with biharmonic inpainting
    large_inpainted = inpaint_biharmonic(img.filled(0), mask=large_glare)

    # now overwrite with these values
    hybrid[large_glare] = large_inpainted[large_glare]

    # put on old image mask
    return ma.masked_array(hybrid, mask=img.mask)


def mask_glare(img, threshold=175, mask_only=False):
    """
    for demoing purposes, with placenta.show_mask

    if mask_only, just return the mask. Otherwise return a copy of img with
    that added to the mask. If you want the original mask to be ignored,
    just pass img.filled(0) ya doofus
    """
    # region to inpaint
    inp = (img > threshold)

    # get a larger area around the specks
    inp = binary_dilation(inp, selem=disk(1))

    # remove anything large
    #inp = white_tophat(inp, selem=disk(3))

    if mask_only:
        return inp
    else:
        # both the original background *and* these new glared regions
        # are masked
        return ma.masked_array(img, mask=inp)


# test it on a particularly bad sample
if __name__ == "__main__":

    from placenta import get_named_placenta, show_mask
    import matplotlib.pyplot as plt

    filename = 'T-BN0204423.png'  # a particularly glary sample
    img = get_named_placenta(filename)

    crop = np.s_[150:500, 150:800]  # indices to zoom in on the region
    zoom = np.s_[300:380, 300:380]  # even smaller region
    masked = mask_glare(img)
    inpainted = inpaint_glare(img)
    minpainted = inpaint_with_boundary_median(img)

    # view the closeup like this
    # save each of these matrices using regular plt.imsave
    minpainted_view = show_mask(minpainted[zoom], interactive=False,
                                mask_color=(103, 15, 23))
    inpainted_view = show_mask(inpainted[zoom], interactive=False,
                               mask_color=(103, 15, 23))
    masked_view = show_mask(masked[zoom], interactive=False,
                            mask_color=(103, 15, 23))
    img_view = show_mask(img[zoom], interactive=False,
                         mask_color=(103, 15, 23))

    # view them all next to each other
    plt.imshow(np.hstack((img_view, masked_view, inpainted_view,
                          minpainted_view)))
