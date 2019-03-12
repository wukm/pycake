#!/usr/bin/env python3

"""
This is extremely obsolete code. This is my filter for fixing the old (slow)
Frangi filter result using a rotating box filter. Here for historical purposes.

given the obsolesence of this code, most external code/ filenames have changed
and this code probably won't work anymore without minor alterations.

If this code is fixed, it should be merged into post_processing.

Alternatively, the Frangi filter+postprocessing techniques demonstrated here
could be added as standalone functions in this module, along with the
'strawman filter' etc.
"""

import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from functools import partial
from skimage.morphology import *
from skimage.exposure import rescale_intensity, equalize_adapthist

from skimage.color import label2rgb

import os
import os.path
import datetime

import numpy as np
import numpy.ma as ma

from skimage.transform import rotate
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.linalg import eig

def rotating_box_filter(img, thetas, sigma, length_ratio=4, verbose=True):
    """
    runs a curvilinear filter at the given scale space `sigma`

    INPUT:
        img:            a binary 2D array
        sigma:          the scale space
        length_ratio:   a rectangular filter will be applied with size
        steps:          the range of rotations (0,180) is divided into this
                        many steps (default: 16 or 12 degrees)
        verbose:        default True

    OUTPUT:
        extracted:  a 2D binary array of the same shape as img

    METHODS:
        (todo)

    IMPLEMENTATION:
        (todo)

    WARNINGS/BUGS:
        this may be supremely wasteful for large step sizes. you should check
        in the anticipated range of sigmas that there is sufficient variation
        in the rotated structure elements to warrant that amount of step sizes.
        print('cleaning up scale space')

        furthermore, this filter should be used carefully. there are probably
        bugs in the logic and implementation.
    """

    sigma = int(sigma) # round down to a integer

    mask = img.mask
    extracted = np.zeros_like(img)
    img = binary_erosion(img, selem=disk(sigma))
    img = binary_dilation(img, selem=disk(sigma))


    width, length = int(2*sigma), int(sigma*length_ratio)

    if length == 0:
        length = 1

    rect = rectangle(width, length)
    outer_rect =  rectangle(int(width+2*sigma+4),int(length))
    outer_rect[sigma:-sigma,:] = 0
    thetas = np.round(thetas*180 / np.pi)

    # this should behave the same as thetas[thetas==180] = 0
    # but not return a warning
    thetas.put(thetas==180, 0) # these angles are redundant

    if verbose:
        print('running vessel_filter with σ={}: w={}, l={}'.format(
            sigma, width,length), flush=True)

    if verbose:
        print('building rotated filters...', end=' ')

    srot = partial(rotate, resize=True, preserve_range=True) # look at order
    rotated = [srot(rect, theta) for theta in range(180)]

    if verbose:
        print('done.')

    if verbose:
        print('building outer filters...', end=' ')
    outer_rotated = [srot(outer_rect, theta) for theta in range(180)]

    for theta in range(180):
        if verbose:
            print('θ=', theta, end='\t', flush=True)
            if theta % 6 == 0:
                print()

        vessels = binary_erosion(img, selem=rotated[theta])
        #margins = binary_dilation(img, selem=outer_rotated[theta])
        #margins = np.invert(margins)
        #vessels = np.logical_and(vessels, margins)
        extracted = np.logical_or(extracted, (thetas == theta) * vessels)
    if verbose:
        print('') # new line

    extracted = binary_dilation(extracted, selem=disk(sigma))
    extracted[mask] = 0
    return extracted

def get_frangi_targets(K1,K2, beta=0.5, c=15, dark_bg=True, threshold=None):
    """
    returns results of frangi filter
    """

    R = (K1/K2) ** 2 # anisotropy
    S = (K1**2 + K2**2) # structureness

    F = np.exp(-R / (2*beta**2))
    F *= 1 - np.exp( -S / (2*c**2))

    if dark_bg:
        F = (K2 < 0)*F
    else:
        F = (K2 > 0)*F

    if threshold:

        return F < threshold
    else:
        return F


def get_targets(K1,K2, method='F', threshold=True):
    """
    returns a binary threshold (conservative)

    F -> frangi filter with default arguments. greater than mean.
    R -> blobness measure. greater than median.
    S -> anisotropy measure (greater than median)
    """
    if method == 'R':
        R = (K1 / K2) ** 2
        if threshold:
            T = R < ma.median(R)
        else:
            T = R
    elif method == 'S':
        S = (K1**2 + K2**2)/2
        if threshold:
            T = S > ma.median(S)
        else:
            T = S
    elif method == 'F':
        R = (K1 / K2) ** 2
        S = (K1**2 + K2**2)/2
        beta, c = 0.5, 15
        F = np.exp(-R / (2*beta**2))
        F *= 1 - np.exp(-S / (2*c**2))
        T = (K2 < 0)*F

        if threshold:
            T = T > (T[T != 0]).mean()
    else:
        raise('Need to select method as "F", "S", or "R"')

    return T

b = partial(plt.imshow, cmap=plt.cm.Blues)
s = plt.show


if __name__ == "__main__":

    #raw = get_preprocessed(mode='G')
    raw = ndi.imread('samples/clahe_raw.png')
    raw = preregister(raw)
    img = preprocess(raw)
    img = raw

    # which σ to use (in order)
    scale_range = np.logspace(0,5, num=30, base=2)

    frangi_only = np.zeros((img.shape[0],img.shape[1],len(scale_range)))
    all_targets = np.zeros((img.shape[0],img.shape[1],len(scale_range)))
    extracted_all = np.zeros((img.shape[0],img.shape[1],len(scale_range)))


    OUTPUT_DIR = 'fpd_new_output'

    n = datetime.datetime.now()

    SUBDIR = ''.join(('', n.strftime('%y%m%d_%H%M')))

    print('saving outputs in', os.path.join(OUTPUT_DIR, SUBDIR))

    try:
        os.mkdir(os.path.join(OUTPUT_DIR, SUBDIR))
    except FileExistsError:
        ans = input('save path already exists! would you like to continue? [y/N]')
        if ans != 'y':
            print('aborting program. clean up after yourself.')

            exit(0)

        else:
            print('your files will be overwritten (but artifacts may remain!)')
    finally:
        print('\n')

    for n, sigma in enumerate(scale_range):

        beta = min(.09*sigma - .04, .5)
        print('-'*80)
        print('σ={}'.format(sigma))

        print('finding hessian')
        h = fft_hessian(img,sigma)

        print('finding curvatures')
        k1,k2 = principal_curvatures(img, sigma=sigma,H=h)

        print('finding targets with β={}'.format(beta))
        t = get_frangi_targets(k1,k2,
                beta=beta, dark_bg=False, threshold=False)
        t = t > t.mean()
        #t = remove_small_objects(t, min_size=100)

        # extend mask
        timg = ma.masked_where(t < t.mean(), img)
        percentage = timg.count() / img.size

        print('finding p. directions {}'.format(np.round(percentage*100)))
        t1,t2 = principal_directions(timg, sigma=sigma, H=h)

        extracted = vessel_filter(t, t1, sigma, length_ratio=.5, verbose=True)

        extracted_all[:,:,n] = extracted

        savefile = ''.join(('%02d' % sigma, '.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile),
                extracted, cmap=plt.cm.Blues)

        all_targets[:,:,n] = (timg!=0) * t1

        #all_targets[:,:,n]= timg!=0

    A = all_targets.sum(axis=-1)
    a = all_targets

    sys.exit(0)

        #new_labels = sigma*np.logical_and(extracted != 0, cumulative == 0)
        #cumulative += new_labels.astype('uint8')

    full_skel = skeletonize(cumulative!=0)
    skel = remove_small_objects(full_skel, min_size=50, connectivity=2)

    matched_all = np.zeros_like(skel)

    for i, scale in enumerate(scale_range):

        e = extracted_all[:,:,i]
        el, nl = label(e, return_num=True)
        matched = np.zeros_like(matched_all)

        for region in range(1, nl+1):
            if np.logical_and(el==region, skel).any():
                matched = np.logical_or(matched, el==region)

        matched_all = np.logical_or(matched_all, matched)


