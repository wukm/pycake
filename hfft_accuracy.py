#!/usr/bin/env python3

"""
visual/tabular demonstration of how different implementations of
calculating the hessian filter compare
"""

BOILERPLATE

show that gaussian blur of hfft is accurate, except potentially around the
boundary proportional to sigma.

or if they're off by a scaling factor, show that the derivates
(taken the same way) are proportional.

pseudocode

A = gaussian_blur(image, sigma, method='convential')
B = gaussian_blur(image, sigma, method='fourier')

zero_order_accurate = isclose(A, B, tol)

J_A= get_jacobian(A)
J_B = get_jacobian(B)

first_order_accurate = isclose(J_A, J_B, tol)

A_eroded = zero_around_plate(A, sigma)
B_eroded = zero_around_plate(B, sigma)

J_A_eroded = zero_around_plate(A, sigma)
J_B_eroded = zero_around_plate(B, sigma)

zero_order_accurate_no_boundary = isclose(A_eroded, B_eroded, tol)
first_order_accurate = isclose(J_A_eroded, J_B_eroded, tol)

"""

from placenta import get_named_placenta, cropped_args

from itertools import combinations_with_replacement
from skimage.exposure import rescale_intensity

from hfft import fft_hessian, fft_gaussian, fft_dgk
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from placenta import show_mask, list_by_quality

from scoring import mean_squared_error
from itertools import combinations
import numpy as np
from scipy.ndimage import laplace
import numpy.ma as ma

from skimage.segmentation import find_boundaries
from skimage.morphology import disk, binary_dilation

from plate_morphology import dilate_boundary

from diffgeo import principal_curvatures
from frangi import structureness, anisotropy, get_frangi_targets

from skimage.util import img_as_float

def plot_image_slices(arrs, fixed_axis=0, fixed_index=None, labels=None,
                      formats=None, title=None):
    """
    arrs needs to be the same shape and dimension
    could pass it to np.stack and check for a value error?

    """
    fig, ax = plt.subplots(figsize=(12,2))
    # hopefully the fixed axis is 0 or 1. this gets the other one
    it_axis = 1 if fixed_axis==0 else 0

    # if it's a tuple, make it an array, etc. etc.
    arrs = np.stack(arrs)

    # make sure we can iterate over it if there's just as single image
    if arrs.ndim < 3:
        arrs = np.expand_dims(arrs,0)

    if labels is None:
        labels = [None for a in arrs]
    if formats is None:
        formats = ['' for a in arrs]

    if fixed_index is None:
        # find halfway point of the appropriate dimension from the first array
        fixed_index = arrs[0].shape[fixed_axis] // 2

    for a, lab, fmt in zip(arrs, labels, formats):
        ax.plot(np.arange(a.shape[it_axis]),
                 np.moveaxis(a, fixed_axis, 0)[fixed_index, :],
                 fmt, label=lab)

    if title is not None:
        ax.set_title(title)

    # can this be at least a little object-oriented? :(
    fig.legend()

def multiway_comparison(arrs, scorefunc):

    scores = np.zeros((len(arrs), len(arrs)))

    for j in range(len(arrs)):
        for k in range(j+1,len(arrs)):
            scores[j,k] = scorefunc(arrs[j], arrs[k])

    return scores

filename = list_by_quality(0)[5]

img = get_named_placenta(filename)

# so that scipy.ndimage.gaussian_filter doesn't use uint8 precision (jesus)
img = ma.masked_array(img_as_float(img), mask=img.mask)

test_sigmas = [.3, .6, 1.0, 5.0, 15, 30, 60, 90]

for sigma in test_sigmas:

    print("*"*80, '\n\n', f"σ={sigma}")
    #print('applying standard gauss blur')

    # this is exactly how it's passed to skimage.feature.hessian_matrix(...)
    A = gaussian_filter(img.filled(0), sigma, mode='constant', cval=0)

    #print('applying fft gauss blur')
    B = fft_gaussian(img, sigma, kernel='sampled')
    C = fft_gaussian(img, sigma, kernel='discrete')

    #print('calculating first derivatives')
    # zero the masks before calculating derivates if they're masked
    Agrad = np.gradient(A)
    Bgrad = np.gradient(B)
    Cgrad = np.gradient(C)


    axes = range(img.ndim)

    #print('calculating second derivatives')
    # this is the same way it's done in skimage.feature.hessian_matrix(...)
    H_A = [np.gradient(Agrad[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)]
    H_B = [np.gradient(Bgrad[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)]
    H_C = [np.gradient(Cgrad[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)]

    #print('calculating eigenvalues of hessian')
    ak1, ak2 = principal_curvatures(img, sigma=sigma, H=H_A)
    bk1, bk2 = principal_curvatures(img, sigma=sigma, H=H_B)
    ck1, ck2 = principal_curvatures(img, sigma=sigma, H=H_C)


    #RA = anisotropy(ak1,ak2)
    #RB = anisotropy(bk1,bk2)
    #RC = anisotropy(ck1,ck2)

    #SA = structureness(ak1, ak2)
    #SB = structureness(bk1, bk2)
    #SC = structureness(ck1, ck2)

    ## ugh, apply masks here. too large to be conservative?
    ## otherwise structureness only shows up for small sizes
    new_mask = dilate_boundary(None, radius=int(3*sigma), mask=img.mask)

    crop = cropped_args(img)

    A = A[crop]
    B = B[crop]
    C = C[crop]

    ak1 = ma.masked_array(ak1,new_mask)[crop]
    ak2 = ma.masked_array(ak2,new_mask)[crop]
    bk1 = ma.masked_array(bk1,new_mask)[crop]
    bk2 = ma.masked_array(bk2,new_mask)[crop]
    ck1 = ma.masked_array(ck1,new_mask)[crop]
    ck2 = ma.masked_array(ck2,new_mask)[crop]

    FA = get_frangi_targets(ak1,ak2, dark_bg=False).filled(0)
    FB = get_frangi_targets(bk1,bk2, dark_bg=False).filled(0)
    FC = get_frangi_targets(ck1,ck2, dark_bg=False).filled(0)


    # the following shows a random vertical slice of A & B (when scaled)
    labels = ('scipy.ndimage,gaussian_filter', 'fft_gaussian', 'fft_dgk')
    formats = ('g:', 'k', 'b-.')
    plot_image_slices((A,B,C), labels=labels, formats=formats,
                      title=r'gaussian convolution $\sigma={}$'.format(sigma))
    plt.tight_layout()
    plt.savefig('Gslice_sigma={:d}.png'.format(int(sigma*10)), dpi=300)
    plot_image_slices((FA,FB,FC), labels=labels, formats=formats,
                      title=r'Frangi filter response $\sigma={}$'.format(sigma))
    plt.tight_layout()
    plt.savefig('Fslice_sigma={:d}.png'.format(int(sigma*10)), dpi=300)
    #plt.show()

    print('comparing gaussians (mean squared error)')
    print(multiway_comparison((A,B,C), mean_squared_error))
    print('comparing frangi response (mean squared error)')
    print(multiway_comparison((FA,FB,FC), mean_squared_error))
