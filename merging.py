#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
from scipy.ndimage import label
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt

def nz_percentile(A, q, axis=None, interpolation='linear'):
    """calculate np.percentile(...,q) on an array's nonzero elements only

    Parameters
    ----------
    A : ndarray
        matrix from which percentiles will be calculated. Percentiles
        are calculated on an elementwise basis, so the shape is not important
    q : a float
        Percentile to compute, between 0 and 100.0 (inclusive).

    (other arguments): see numpy.percentile docstring
    ...

    Returns
    -------
    out: float

    """

    if ma.is_masked(A):
        A = A.filled(0)

    return np.percentile(A[A > 0], q, axis=axis, interpolation=interpolation)


def apply_threshold(targets, alphas, return_labels=True):
    """Threshold targets at each scale, then return max target over all scales.

    A unique alpha can be given for each scale (see below). Return a 2D boolean
    array, and optionally another array representing what at what scale the max
    filter response occurred.

    Parameters
    ----------
    targets : ndarray
        a 3D array, where targets[:,:,k] is the result of the Frangi filter
        at the kth scale.
    alphas : float or array_like
        a list / 1d array of length targets.shape[-1]. each alphas[k] is a
        float which thresholds the Frangi response at the kth scale. Due to
        broadcasting, this can also be a single float, which will be applied
        to each scale.
    return_labels : bool, optional
        If True, return another ndarray representing the scale (see Notes
        below). Default is True.

    Returns
    -------
    out : ndarray, dtype=bool
        if return labels is true, this will return both the final
        threshold and the labels as two separate matrices. This is
        a convenience, since you could easily find labels with
    labels : ndarray, optional, dtype=uint8
        The scale at which the largest filter response was found after
        thresholding. Element is 0 if no scale passed the threshold,
        otherwise an int between 1 and targets.shape[-1] See Notes below.

    Notes / Examples
    ----------------
    Despite the name, this does *NOT* return the thresholded targets itself,
    but instead the maximum value after thresholding. If you wanted the
    thresholded filter responses alone, you should simply run

    >>>(targets > alphas)*targets

    The optional output `labels` is a 2D matrix indicating where the max filter
    response occured. For example, if the label is K, the max filter response
    will occur at targets[:,:,K-1].  In other words,

    >>>passed, labels = apply_threshold(targets,alphas)
    >>>targets.max(axis=-1) == targets[:,:,labels -1 ]
    True

    It should be noted that returning labels is really just for convenience
    only; you could construct it as shown in the following example:

    >>>manual_labels = (targets.argmax(axis=-1) + 1)*np.invert(passed)
    >>>labels == manual_labels
    True

    Similarly, the standard boolean output could just as easily be obtained.
    >>>passed == (labels != 0)
    True
    """

    # threshold as an array (even if it's a single element) to broadcast
    alphas = np.array(alphas)

    # if input's just a MxN matrix, expand it trivially so it works below
    if targets.ndim == 2:
        targets = np.expand_dims(targets, 2)

    # either there's an alpha for each channel or there's a single
    # alpha to be broadcast across all channels
    assert (targets.shape[-1] == alphas.size) or (alphas.size == 1)

    # pixels that passed the threshold at any level
    passed = (targets >= alphas).any(axis=-1)

    if not return_labels:
        return passed  # we're done already

    wheres = targets.argmax(axis=-1)  # get label of where maximum occurs
    wheres += 1  # increment to reserve 0 label for no match

    # then remove anything that didn't pass the threshold
    wheres[np.invert(passed)] = 0

    assert np.all(passed == (wheres > 0))

    return passed, wheres


def sieve_scales(multiscale, high_percentile, low_percentile, min_size=None,
                 axis=0):
    """
    multiscale is a 3 dimensional where 2 dimensions are image and `axis`
    parameter is which one is the scale space (i.e. resolution). hopefully
    axis is 0 or 1 (this won't handle stupider cases)

    this gathers points contiguous points at a low threshold and adds them
    to the output it contains at least only if that blob contains at least one
    high percentile point.

    min_size is a size requirement can either be an integer or an array of
    integers """

    assert multiscale.ndim == 3

    if axis in (-1, 2):
        # this won't change the input, just creates a view
        V = np.transpose(multiscale, axes=(2, 0, 1))

    elif axis == 0:
        V = multiscale  # just to use the same variable name
    else:
        raise ValueError('Please make resolution the first or last dimension.')

    if np.isscalar(min_size):
        min_size = [min_size for x in range(multiscale.shape[0])]

    # label matrix the size of one of the images
    sieved = np.zeros(V.shape[1:], dtype=np.int32)


    for n, v in enumerate(V):
        if min_size is not None:
            z = remove_small_objects(v, min_size=min_size[n])
        else:
            z = v  # relabel to use same variable

        high_thresh = nz_percentile(v, high_percentile)
        low_thresh = nz_percentile(v, low_percentile)

        labeled, n_labels = label(z > low_thresh)
        high_passed = (z > high_thresh)

        for lab in range(n_labels):
            if lab == 0:
                continue
            if np.any(high_passed[labeled == lab]):
                sieved[labeled == lab] = n

    return sieved


def view_slices(multiscale, axis=0, scales=None, cmap='nipy_spectral',
                vmin=0, vmax=1.0, outnames=None, show_colorbar=True):
    """ scales is just to use for a figure title
        crop before you get in here.

        if outname is an iterable returning filenames, then we'll assume
        non-interative mode
        """
    assert multiscale.ndim == 3

    if axis in (-1, 2):
        # this won't change the input, just creates a view
        V = np.transpose(multiscale, axes=(2, 0, 1))

    elif axis == 0:
        V = multiscale  # just to use the same variable name
    else:
        raise ValueError('Please make resolution the first or last dimension.')

    if scales is None:
        scales = [None for x in range(multiscale.shape[0])]
    if outnames is None:
        outnames = [None for x in range(multiscale.shape[0])]

    for v, sigma, outname in zip(V, scales, outnames):

        if outname is None:
            plt.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.tight_layout()
            if sigma is not None:
                plt.title(r'$\sigma={:.2f}$'.format(sigma))
            plt.axis('off')
            if show_colorbar:
                plt.colorbar()
            plt.tight_layout()
            plt.show()
            plt.close()

        else:
            # save them non interactively with imsave
            plt.imsave(outname, v, cmap=cmap, vmin=vmin, vmax=vmax)

