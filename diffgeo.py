#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.linalg import eig


def principal_curvatures(img, sigma=1.0, H=None):
    """Calculate the approximated principal curvatures of an image

    Return the (approximated) principal curvatures {κ1, κ2} of an image,
    that is, the eigenvalues of the Hessian at each point (x,y). The output
    is arranged such that |κ1| <= |κ2|.  Note that the Hessian of the image,
    if not provided, is computed using skimage.feature.hessian_matrix, which
    can be very slow for large sigmas.

    Parameters
    ----------
    img: array or ma.MaskedArray

        An ndarray representing a 2D or multichannel image. If the image is
        multichannel (e.g. RGB), then each channel will be proccessed
        individually. Additionally, the input image may be a masked array-- in
        which case the output will preserve this mask identically.

    sigma: float, optional
        Standard deviation of the Gaussian (used to calculate the hessian
        matrix).
    H:  list of array, optional
        The hessian itself (Hxx,Hxy,Hyy) whose eigenvalues will be calculated.
        Use this option if you're going to calculate the Hessian using faster
        means, e.g. via FFT.

    Returns
    -------
    (K1, K2):   tuple of arrays
        K1, K2 each are the exact dimension of the input image, ordered in
        magnitude such that |κ1| <= |κ2| in all locations.

    Examples
    --------
        >>> K1, K2 = principal_curvatures(img)
        >>> K1.shape == img.shape
        True
        >>> (K1 <= K2).all()
        True
        >>> K1.mask == img.mask
        True
    """

    # determine if multichannel
    multichannel = (img.ndim == 3)

    if not multichannel:
        # add a trivial dimension
        img = img[:,:,np.newaxis]

    K1 = np.zeros_like(img, dtype='float64')
    K2 = np.zeros_like(img, dtype='float64')

    for ic in range(img.shape[2]):

        channel = img[:,:,ic]

        # returns the tuple (Hxx, Hxy, Hyy)
        if H is None:
            H = hessian_matrix(channel, sigma=sigma)

        # returns tuple (l1,l2) where l1 >= l2 but this *includes sign*
        L = hessian_matrix_eigvals(H)
        L = reorder_eigs(L)

        # Make K2 larger in magnitude, as consistent with Frangi paper
        K1[:, :, ic] = L[0, :, :]
        K2[:, :, ic] = L[1, :, :]

    try:
        mask = img.mask  # get mask to add to each if input was a masked array

    except AttributeError:

        pass  # there's no mask, so do nothing

    else:
        K1 = ma.masked_array(K1, mask=mask)
        K2 = ma.masked_array(K2, mask=mask)

    # now undo the trivial dimension
    if not multichannel:
        K1 = np.squeeze(K1)
        K2 = np.squeeze(K2)

    return K1, K2


def reorder_eigs(L):
    """reorder eigenvalues by decreasing magnitude.

    Eigenvalues are outputted from hessian_matrix_eigvals so that L1 >= L2.
    This reorders this so that |L1| >= |L2| instead (where L1,L2=L)
    Parameters
    ----------
    L:  ndarray or iterable of ndarrays
        As outputted by, say, hessian_matrix_eigs. If a single ndarray, it
        should be the shape (N, *img.shape) where there are N eigenvalues to
        reorder. You may also input a tuple like (L1,L2).
    Returns
    -------
    eigs: ndarray
        The eigenvalues in decreasing order of magnitude; that is
        eigs[i,j,k] is the ith-largest eigenvalue at position (j, k).
        Each of these is the same shape the original inputs, but
        np.abs(L1r) >= np.abs(L2r) will be true. See warning below.


    Warnings / Notes
    ----------------
    Please note the order! Outputs are given in *decreasing* magnitude. This is
    done to align with the behavior of skimage.feature.hessian_matrix_eigvals,
    but if you want to label them according to the Frangi filter (where k2
    denotes the *larger* magnitude eigenvalue, you should reverse the labels:

        >>>k2, k1 = reorder_eigs(L)  # k2, k1 as frangi labeled them
        >>>np.all(np.abs(k2) >= np.abs(k1))
        True

    It doesn't actually matter the order in which inputs are inputted (they
    will be sorted the same regardless).

    Example
    -------
    >>>K1,K2 = hessian_matrix_eigvals(H)
    >>>(K1 >= K2).all()
    True
    >>>(np.abs(K1) <= np.abs(K2)).all()
    False
    >>>K1r, K2r = reorder_eigs(K1,K2)
    >>>(K1r <= K2r).all()
    False
    >>>(np.abs(K1r) <= np.abs(K2r)).all()
    True

    TODO
    ----
    Support out= keyword
    """
    # this will do nothing if L is already an array but will make it an array
    # if it's a tuple/list/iterable
    L = np.stack(L)
    mag =  np.argsort(np.abs(L), axis=0)

    # now L2 is larger in absolute value, as consistent with Frangi paper
    return np.take_along_axis(L, mag, axis=0)


def principal_directions(img, sigma, H=None, mask=None):
    """Calculate principal directions of
    will ignore calculation of principal directions of masked areas

    mask should  be positive where the PD's should *NOT* be calculated
    this function actually returns the theta corresponding to
    leading and trailing principal directions, i.e. angle w / x axis
    """

    if H is None:
        H = hessian_matrix(img, sigma)

    Hxx, Hxy, Hyy = H


    # determine if there was a supplied mask or use images if it exists
    if mask is None:
        try:
            mask = img.mask
        except AttributeError:
            masked = False
        else:
            masked = True
    else:
        masked = True

    dims = img.shape

    # where to store
    trailing_thetas = np.zeros_like(img, dtype='float64')
    leading_thetas = np.zeros_like(img, dtype='float64')

    # maybe implement a small angle correction
    for i, (xx, xy, yy) in enumerate(np.nditer([Hxx, Hxy, Hyy])):

        # grab the (x,y) coordinate of the hxx, hxy, hyy you're using
        subs = np.unravel_index(i, dims)

        # ignore masked areas (if masked array)
        if masked and mask[subs]:
            continue

        h = np.array([[xx, xy], [xy, yy]])  # per-pixel hessian
        l, v = eig(h)  # eigenvectors as columns

        # reorder eigenvectors by (increasing) magnitude of eigenvalues
        v = v[:, np.argsort(np.abs(l))]

        # angle between each eigenvector and positive x-axis
        # arccos of first element (dot product with (1,0) and eigvec is already
        # normalized)
        trailing_thetas[subs] = np.arccos(v[0, 0])  # first component of each
        leading_thetas[subs] = np.arccos(v[0, 1])  # first component of each

    if masked:
        leading_thetas = ma.masked_array(leading_thetas, mask)
        trailing_thetas = ma.masked_array(trailing_thetas, mask)

    return trailing_thetas, leading_thetas


if __name__ == "__main__":

    pass

    #from get_base import get_preprocessed
    #import matplotlib.pyplot as plt
    #from functools import partial
    #from fpd import get_targets
    #b = partial(plt.imshow, cmap=plt.cm.Blues)
    #sp = partial(plt.imshow, cmap=plt.cm.spectral)
    #s = plt.show

    #import time

    #img = get_preprocessed(mode='G')

    #for sigma in [0.5, 1, 2, 3, 5, 10]:

    #    print('-'*80)
    #    print('σ=',sigma)
    #    print('calculating hessian H')

    #    tic = time.time()
    #    H = hessian_matrix(img, sigma=sigma)

    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()
    #    print('calculating hessian via FFT (F)')
    #    h = fft_hessian(img, sigma)

    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()
    #    print('calculating principal curvatures for σ={}'.format(sigma))
    #    K1,K2 = principal_curvatures(img, sigma=sigma, H=H)
    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()
    #    print('calculating principal curvatures for σ={} (fast)'.format(sigma))
    #    k1,k2 = principal_curvatures(img, sigma=sigma, H=h)

    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()

    #    #####

    #    print('calculating targets for σ={}'.format(sigma))
    #    T = get_targets(K1,K2, threshold=False)

    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()

    #    print('calculating targets for σ={} (fast)'.format(sigma))
    #    t = get_targets(k1,k2, threshold=False)

    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)

    #    ######

    #    print('extending masks')

    #    # extend mask over nontargets items
    #    img1 = ma.masked_where( T < T.mean(), img)
    #    img2 = ma.masked_where( t < t.mean(), img)

    #    tic = time.time()
    #    print('calculating principal directions for σ={}'.format(sigma))
    #    T1,T2 = principal_directions(img1, sigma=sigma, H=H)
    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
    #    tic = time.time()

    #    print('calculating principal directions for σ={} (fast)'.format(sigma))
    #    t1,t2 = principal_directions(img2, sigma=sigma, H=h)
    #    toc = time.time()
    #    print('time elapsed: ', toc - tic)
