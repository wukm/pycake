#!/usr/bin/env python3
##TODO: - just one confusion function
##      - allow to change colors for confusion (dict outside of function)
##      - move color to width function here from get_placenta
##      - MCCS include outside plate if keyword passed

import numpy as np
from get_placenta import open_typefile, open_tracefile

def get_widths_from_trace(T, min_width=3, max_width=19, widths=None):
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

        min_width: widths below this will be excluded (default is
                    3, the min recorded width). assuming these
                    are ints
        max_width: widths above this will be excluded (default is
                    19, the max recorded width)

        widths: an explicit list of widths that should be returned.
                in this case the above min & max are ignored.
                this way you could include widths = [3, 17, 19] only

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

    ##TODO: expand this later to handle arterial traces and venous traces
    """

    # a 2D picture to fix in with the pixel widths
    widthtrace = np.zeros_like(T[:,:,0])

    for pix, color in TRACE_COLORS.items():

        # get the 2D indices that are that color
        idx = np.where(np.all(T == color, axis=-1))
        widthtrace[idx] = pix

    if widths is None:
        min_width, max_width = int(min_width), int(max_width)
        T[T < min_width] = 0
        T[T > max_width] = 0
    else:
        # use numpy.isin(T, widths) but that's only in
        # version 1.13 and up of numpy

        # elements in A that can be found in
        # need to reshape, after v.1.13 of numpy you can use np.isin
        to_keep = np.in1d(T,x,assume_unique=True).reshape(A.shape)

        T[np.invert(to_keep)] = 0

    if as_binary:
        return T != 0
    else:
        return T

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

def widths_to_colors(w, show_non_matches=False):
    """
    FOR DISPLAY PURPOSES / convenience

    return an RGB matrix of ints [0,255] converting back from
    [3,5,7, ... , 19] -> TRACE_COLORS

    actually making a matplotlib colormap didn't seem worth it

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
    return tuple(int(x,16) for x in (triple[:2],triple[2:4],triple[4:]))

def confusion(test, truth, colordict=None):
    """
    distinct coloration of false positives and negatives.

    colors output matrix with
    true_pos if test[-] == truth[-] == 1
    true_neg if test[-] == truth[-] == 0
    false_neg if test[-] == 0 and truth[-] == 1
    false_pos if test[-] == 1 and truth[-] == 0

    if colordict is supplied: you supply a dictionary of how to
    color the four cases. Spec given by the default below:

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
            'TN': (247, 247, 247), # true negative# 'f7f7f7'
            'TP': (0, 0, 0), # true positive  # '000000'
            'FN': (241, 163, 64), # false negative # 'f1a340' orange
            'FP': (153, 142, 195), # false positive # '998ec4' purple
            'mask': (247, 200, 200) # mask color (not used in MCC calculation)
            }

    # TODO: else check if mask is specified and add it as color of TN otherwise

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

    # color the mask !!

    return output

def compare_trace(approx, trace=None, tracefile=None, filename=None,
                  sample_dir=None, a_color=None, b_color=None):
    """
    compare approx matrix to trace matrix and output a confusion matrix.
    if trace is not supplied, open the image from the tracefile.
    if tracefile is not supplied, filename must be supplied, and
    tracefile will be opened according to the standard pattern.

    a_color and b_color are parameters to pass to confusion()

    returns a matrix
    """

    # load the tracefile if not supplied
    if trace is None:
        if tracefile is not None:
            trace = open_tracefile(filename)
        elif filename is not None:
            try:
                trace = open_typefile(filename, 'trace')
            except FileNotFoundError:
                print("No trace file found matching ", filename)
                print("no trace found. generating dummy trace.")
                trace = np.zeros_like(approx)
        else:
            print("no trace supplied/found. generating dummy trace.")
            trace = np.zeros_like(approx)

    # what a mess... trace should be inverted (black is BG)
    trace = np.invert(trace)

    # calculate the confusion matrix
    # assert same size and dimension?
    #C = confusion(approx, trace, a_color, b_color)
    C = confusion_4(approx,trace)
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
    Passing score_bg=True makes this decision explicit.
    (Check this)

    """
    true_pos = np.bitwise_and(test==truth, truth)
    true_neg = np.bitwise_and(test==truth, np.invert(truth))
    false_neg = np.bitwise_and(truth, np.invert(test))
    false_pos = np.bitwise_and(test, np.invert(truth))

    if score_bg or (bg_mask is not None):
        # only get stats in the plate
        true_pos[bg_mask] = 0
        true_neg[bg_mask] = 0
        false_pos[bg_mask] = 0
        false_neg[bg_mask] = 0

    TP = true_pos.sum()
    TN = true_neg.sum()
    FP = false_pos.sum()
    FN = false_neg.sum()
    total = np.invert(bg_mask).sum()
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


