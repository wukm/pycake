#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma


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


