#!/usr/bin/env python3

import numpy as np
from placenta import open_typefile, open_tracefile
from skimage.morphology import thin

def rgb_to_widths(T):
    """
    this will take an RGB trace image (MxNx3) and return a 2D (MxN)
    "labeled" trace corresponding to the traced pixel length.
    there is no distinguishing between arteries and vessels

    it's preferrable to do this in real-time so only one tracefile
    needs to be stored (making the sample folder less cluttered)
    although obviously at the expense of storing a larger image
    which is only needed for visualization purposes.

    Input:
        T: a MxNx3 RGB (uint8) array, where the colorations are
        assumed as described in NOTES below.

    Output:
        widthtrace: a MxN array whose inputs describe the width of the
        vessel (in pixels), see NOTES.

    Notes:

        The correspondence is as follows:
        3 pixels: "#ff006f",  # magenta
        5 pixels: "#a80000",  # dark red
        7 pixels: "#a800ff",  # purple
        9 pixels: "#ff00ff",  # light pink
        11 pixels: "#008aff",  # blue
        13 pixels: "#8aff00",  # green
        15 pixels: "#ffc800",  # dark yellow
        17 pixels: "#ff8a00",  # orange
        19 pixels: "#ff0015"   # bright red

    According to the original tracing protocol, the traced vessels are
    binned into these 9 sizes. Vessels with a diameter smaller than 3px
    are not traced (unless they're binned into 3px).

    Note: this does *not* deal with collisions. If you pass anything
    with addition (blended colors) as the ctraces are, you will have
    trouble, as those will not be registered as any of the colors above
    and will thus be ignored. If you want to handle data from both
    arterial *and* venous layers, you should do so outside of this
    function.
    """

    # a 2D picture to fix in with the pixel widths
    W = np.zeros_like(T[:,:,0])

    for pix, color in TRACE_COLORS.items():

        #ignore pixelwidths outside the specified range
        # get the 2D indices that are that color
        idx = np.where(np.all(T == color, axis=-1))
        W[idx] = pix


    return W

def merge_widths_from_traces(A_trace, V_trace, strategy='minimum'):
    """
    combine the widths from two RGB-traces A_trace and V_trace
    and return one width matrix according to `strategy`

    Parameters
    ----------
    A_trace: ndarray
        an MxNx3 matrix, where each pixel (along the
        last dimension) is an RGB triplet (i.e. each entry
        is an integer between [0,256). The colors each
        correspond to those in TRACE_COLORS, and (255,255,255)
        signifies "no vessel". This will normally correspond to
        the sample's arterial trace.
    V_trace: ndarray
        an MxNx3 matrix the same shape and other
        requirements as A_trace (see above). This will normally
        correspond to the sample's venous trace.
    strategy: keyword string
        when A_trace and V_trace coincide at some entry,
        this is the merging strategy. It should be a keyword
        of one of the following choices:

        "minimum": take the minimum width of the two traces
            (default). this is the sensible option if you
            are filtering out larger widths.
        "maximum": take the maximum width of the two traces
        "artery" or "A" or "top": take the width from A_trace
        "vein" or "V" or "bottom": take the width from V_trace

    Returns
    -------
        W : ndarray
        a width-matrix where each entry is a number 0 (no vessel), 3,5,7,...19

    Notes
    -----
    Since arteries grow over the veins on the PCSVN and are generally easier
    to extract, it might be preferable to indicate "arteries". In reality,
    each strategy is a compromise, and only by keeping track of both would
    you get the complete picture.

    No filtering out widths is done here.
    """
    assert A_trace.shape == V_trace.shape

    A = rgb_to_widths(A_trace)
    V = rgb_to_widths(V_trace)

    # collisions (where are widths both reported)
    c = np.logical_and(A!=0, V!=0)

    W = np.maximum(A,V)  # get the nonzero value
    if strategy == 'maximum':
        pass # already done, else rewrite the collisions
    elif strategy in ('arteries', 'A', 'top'):
        W[c] = A[c]
    elif strategy in ('veins', 'V', 'bottom'):
        W[c] = V[c]
    else:
        if strategy != 'minimum':
            print(f"Warning: unknown merge strategy: {strategy}")
            print("Defaulting to minimum strategy")

        W[c] = np.minimum(A[c], V[c])

    return W

def filter_widths(W, widths=None, min_width=3, max_width=19):
    """
    Filter a width matrix, removing widths according to rules.

    This function will take a 2D matrix of vessel widths and
    remove any widths outside a particular range (or alternatively,
    that are not included in a particular list)

    Should be roughly as easy as doing it by hand, except that you
    won't have to rewrite the code each time.

    Inputs:

    W:  a width matrix (2D matrix with elements 0,3,5,7,...19

    min_width: widths below this will be excluded (default is
                3, the min recorded width). assuming these
                are ints

    max_width: widths above this will be excluded (default is
                19, the max recorded width)

    widths: an explicit list of widths that should be returned.
            in this case the above min & max are ignored.
            this way you could include widths = [3, 17, 19] only
            """

    Wout = W.copy()
    if widths is None:
        Wout[W < min_width] = 0
        Wout[W > max_width] = 0

    else:
        # use numpy.isin(T, widths) but that's only in version 1.13 and up
        # of numpy this is basically the code for that though
        to_keep = np.in1d(W, widths, assume_unique=True).reshape(W.shape)
        Wout[np.invert(to_keep)] = 0
    return Wout


TRACE_COLORS = {
    3: (255, 0, 111),
    5: (168, 0, 0),
    7: (168, 0, 255),
    9: (255, 0, 255),
    11: (0, 138, 255),
    13: (138, 255, 0),
    15: (255, 200, 0),
    17: (255, 138, 0),
    19: (255, 0, 21)
}


def widths_to_rgb(w, show_non_matches=False):
    """Convert width matrix back to RGB values.

    For display purposes/convenience. Return an RGB matrix
    converting back from [3,5,7, ... , 19] -> TRACE_COLORS

    this doesn't do any rounding (i.e. it ignores anything outside of
    the default widths), but maybe you'd want to?
    """
    B = np.zeros((w.shape[0], w.shape[1], 3))

    for px, rgb_triplet in TRACE_COLORS.items():
        B[w == px, : ] = rgb_triplet

    if show_non_matches:
        # everything in w not found in TRACE_COLORS will be black
        B[w == 0, : ] = (255, 255, 255)
    else:
        non_filled = (B == 0).all(axis=-1)

        B[non_filled,:] = (255,255,255) # make everything white

    # matplotlib likes the colors as [0,1], so....
    return B / 255.


def _hex_to_rgb(hexstring):
    """
    there's a function that does this in matplotlib.colors
    but its scaled between 0 and 1 but not even as an
    array so this is just as much work

    ##TODO rewrite everything so this is useful if it's not been
    rewritten already.
    """
    triple = hexstring.strip("#")
    return tuple(int(x, 16) for x in (triple[:2], triple[2:4], triple[4:]))


def skeletonize_trace(T, T2=None):
    """
    if T is a boolean matrix representing a trace, then thin it

    if T is an RGB trace, then register it according to the
    tracing protocol then thin it

    if T2 is provided, do the same thing to T2 and then merge the two
    """
    if T.ndim == 3:
        trace = (rgb_to_widths(T) > 0)  # booleanize it

    thinned = thin(trace)

    if T2 is None:
        return thinned

    else:
        # do the same thing to second trace and merge it
        if T2.ndim == 3:
            trace_2 = (rgb_to_widths(T2) > 0)  # booleanize it
        thinned_2 = thin(trace_2)

        return np.logical_or(thinned, thinned_2)


def confusion(test, truth, bg_mask=None, colordict=None, tint_mask=True):
    """
    distinct coloration of false positives and negatives.

    colors output matrix with
        true_pos if test[-] == truth[-] == 1
        true_neg if test[-] == truth[-] == 0
        false_neg if test[-] == 0 and truth[-] == 1
        false_pos if test[-] == 1 and truth[-] == 0

    if colordict is supplied: you supply a dictionary of how to
    color the four cases. Spec given by the default below:

    if tint mask, then the mask is overlaid on the image, not replacing totally
    colordict = {
        'TN': (247, 247, 247), # true negative
        'TP': (0, 0, 0) # true positive
        'FN': (241, 163, 64), # false negative
        'FP': (153, 142, 195), # false positive
        'mask': (247, 200, 200) # mask color (not used in MCC calculation)
        }
    """

    if colordict is None:
        colordict = {
            'TN': (247, 247, 247),  # true negative# 'f7f7f7'
            'TP': (0, 0, 0),  # true positive  # '000000'
            'FN': (241, 163, 64),  # false negative # 'f1a340' orange
            'FP': (153, 142, 195),  # false positive # '998ec4' purple
            'mask': (247, 200, 200)  # mask color (not used in MCC calculation)
            }

    #TODO: else check if mask is specified and add it as color of TN otherwise

    true_neg_color = np.array(colordict['TN'], dtype='f')/255
    true_pos_color = np.array(colordict['TP'], dtype='f')/255
    false_neg_color = np.array(colordict['FN'], dtype='f') /255
    false_pos_color = np.array(colordict['FP'], dtype='f')/255
    mask_color = np.array(colordict['mask'], dtype='f') /255

    assert test.shape == truth.shape

    # convert to bool
    test, truth = test.astype('bool'), truth.astype('bool')

    # RGB array size of test and truth for output
    output = np.zeros((test.shape[0], test.shape[1], 3), dtype='f')

    # truth conditions
    true_pos = np.bitwise_and(test==truth, truth)
    true_neg = np.bitwise_and(test==truth, np.invert(truth))
    false_neg = np.bitwise_and(truth, np.invert(test))
    false_pos = np.bitwise_and(test, np.invert(truth))

    output[true_pos,:] = true_pos_color
    output[true_neg,:] = true_neg_color
    output[false_pos,:] = false_pos_color
    output[false_neg,:] = false_neg_color

    # try to find a mask
    if bg_mask is None:
        try:
            bg_mask =  test.mask
        except AttributeError:
            # no mask is specified, we're done.
            return output

    # color the mask
    if tint_mask:
        output[bg_mask,:] += mask_color
        output[bg_mask,:] /= 2
    else:
        output[bg_mask,:] = mask_color

    return output


def compare_trace(approx, trace=None, filename=None,
                  sample_dir=None, colordict=None):
    """
    compare approx matrix to trace matrix and output a confusion matrix.
    if trace is not supplied, open the image from the tracefile.
    if tracefile is not supplied, filename must be supplied, and
    tracefile will be opened according to the standard pattern

    colordict are parameters to pass to confusion()

    returns a matrix
    """

    # load the tracefile if not supplied
    if trace is None:
        if filename is not None:
            try:
                trace = open_typefile(filename, 'trace')
            except FileNotFoundError:
                print("No trace file found matching ", filename)
                print("no trace found. generating dummy trace.")
                trace = np.zeros_like(approx)
        else:
            print("no trace supplied/found. generating dummy trace.")
            trace = np.zeros_like(approx)

    C = confusion(approx, trace, colordict=colordict)

    return C


def mcc(test, truth, bg_mask=None, score_bg=False, return_counts=False):
    """
    Matthews correlation coefficient
    returns a float between -1 and 1
    -1 is total disagreement between test & truth
    0 is "no better than random guessing"
    1 is perfect prediction

    bg_mask is a mask of pixels to ignore from the statistics
    for example, things outside the placental plate will be counted
    as "TRUE NEGATIVES" when there wasn't any chance of them not being
    scored as negative. therefore, it's not really a measure of the
    test's accuracy, but instead artificially pads the score higher.

    setting bg_mask to None when test and truth are not masked
    arrays should give you this artificially inflated score.
    Passing score_bg=True makes this decision explicit, i.e.
    any masks (even if supplied) will be ignored, and your count of
    false positives will be inflated.

    """
    true_pos = np.logical_and(test==truth, truth)
    true_neg = np.logical_and(test==truth, np.invert(truth))
    false_neg = np.logical_and(truth, np.invert(test))
    false_pos = np.logical_and(test, np.invert(truth))

    if score_bg:
        # take the classifications above as they are (nothing is masked)
        pass
    else:
        # if no specified mask, check the test array itself?
        if bg_mask is None:
            try:
                bg_mask = test.mask
            except AttributeError:
                # no mask is specified, we're done.
                bg_mask = np.zeros_like(test)

        # only get stats in the plate
        true_pos[bg_mask] = 0
        true_neg[bg_mask] = 0
        false_pos[bg_mask] = 0
        false_neg[bg_mask] = 0

    # now tally
    TP = true_pos.sum()
    TN = true_neg.sum()
    FP = false_pos.sum()
    FN = false_neg.sum()

    if not score_bg:
        total = np.invert(bg_mask).sum()
    else:
        total = test.size
    #print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    #print('TP+TN+FN+FP={}\ntotal pixels={}'.format(TP+TN+FP+TN,total))
    # prevent potential overflow
    denom = np.sqrt(TP+FP)*np.sqrt(TP+FN)*np.sqrt(TN+FP)*np.sqrt(TN+FN)

    if denom == 0:
        # set MCC to zero if any are zero
        m_score =  0
    else:
        m_score = ((TP*TN) - (FP*FN)) / denom

    if return_counts:
        return m_score, (TP,TN,FP,FN)
    else:
        return m_score


def mean_squared_error(A,B):
    """
    get mean squared error between two matrices of the same size

    input:
        A, B : two ndarrays of the same size.

    output:

        mse:   a single number.
    """

    try:
        mse = ((A-B)**2).sum() / A.size

    except ValueError:
        print("inputs must be of the same size")
        raise

    return mse


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.data import binary_blobs

    A = binary_blobs()
    B = binary_blobs()

    true_neg_color = np.array([247,247,247], dtype='f') # 'f7f7f7'
    true_pos_color = np.array([0, 0, 0] , dtype='f')  # '000000'
    false_neg_color = np.array([241,163,64], dtype='f')# 'f1a340'
    false_pos_color = np.array([153,142,195], dtype='f') # '998ec4'

    C = confusion(A,B)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                        ncols=3,
                                        figsize=(8, 2.5),
                                        sharex=True,
                                        sharey=True)

    ax0.imshow(A, cmap='gray')
    ax0.set_title('A')
    ax0.axis('off')
    ax0.set_adjustable('box-forced')

    ax1.imshow(B, cmap='gray')
    ax1.set_title('B')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')

    ax2.imshow(C)
    ax2.set_title('confusion matrix of A and B')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')

    fig.tight_layout()


