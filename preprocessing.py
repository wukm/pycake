#!/usr/bin/env python3

from skimage.morphology import binary_dilation, white_tophat, disk
import numpy as np
import numpy.ma as ma

def inpaint_glare(img, threshold=175, window_size=15):
    """
    img is a masked array type uint [0,255]
    """

    # bool array, true where glare
    glared = mask_glare(img, threshold=threshold, mask_only=True)

    B = ma.masked_array(img, mask=glared)  # masked background *and* glare
    new_img = img.copy()  # copy values of orignal image (will rewrite)
    d = int(window_size)

    for j, k in zip(*np.where(glared)):
        # rewrite all glared pixels with the mean of nonmasked elements
        # in a window_size window. (this doesn't check OoB, be careful!)
        new_img[j, k] = B[j-d:j+d, k-d:k+d].compressed().mean()

    return new_img


def mask_glare(img, threshold=185, mask_only=False):
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
    inp = white_tophat(inp, selem=disk(3))

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

    filename = 'T-BN0204423.png' # a particularly glary sample
    img = get_named_placenta(filename)

    crop = np.s_[150:500,150:800] # indices to zoom in on the region
    masked = mask_glare(img)
    inpainted = inpaint_glare(img)

    # view the closeup like this
    # save each of these matrices using regular plt.imsave
    inpainted_view = show_mask(inpainted[crop], interactive=False,
                               mask_color=(103,15,23))
    masked_view = show_mask(masked[crop], interactive=False,
                            mask_color=(103,15,23))
    img_view = show_mask(img[crop], interactive=False,
                            mask_color=(103,15,23))

    # view them all next to each other
    plt.imshow(np.hstack((img_view, masked_view, inpainted_view)))

