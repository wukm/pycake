#!/usr/bin/env python3

import numpy as np
from placenta import open_typefile, open_tracefile
from skimage.morphology import thin

import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
from collections import deque

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
    c = (A!=0)& (V!=0)

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
        Wout[~to_keep] = 0
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
    else:
        trace = T.astype('bool')

    thinned = thin(trace)

    if T2 is None:
        return thinned

    else:
        # do the same thing to second trace and merge it
        if T2.ndim == 3:
            trace_2 = (rgb_to_widths(T2) > 0)  # booleanize it
        thinned_2 = thin(trace_2)

        return np.logical_or(thinned, thinned_2)

def precision(counts):

    return int(t[0]) / int(t[0] + t[2])


def integrate_score(score, truth, mask=None):
    """Integrate/sum a probability-like score over a ground truth subset.
    Truth is a binary matrix
    """

    if mask is None:
        if ma.is_masked(score):
            plate = score.filled(0)
        else:
            plate = score

        ground_truth = truth

    else:
        plate = score*(~mask)
        ground_truth = truth*(~mask)

    subset_sum = score[truth].sum()
    total_sum = score.sum()

    return subset_sum / total_sum


def confusion(test, truth=None, bg_mask=None, colordict=None, tint_mask=True):
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
#        colordict = {
#            'TN': (247, 247, 247),  # true negative# 'f7f7f7'
#            'TP': (0, 0, 0),  # true positive  # '000000'
#            'FN': (241, 163, 64),  # false negative # 'f1a340' orange
#            'FP': (153, 142, 195),  # false positive # '998ec4' purple
#            'mask': (247, 200, 200)  # mask color (not used in MCC calculation)
#            }
#
        #colordict = {
        #    'TN': (49,49,49),  # true negative# 'f7f7f7'
        #    'TP': (0, 0, 0),  # true positive  # '000000'
        #    'FN': (201,53,108),  # false negative # 'f1a340' orange
        #    'FP': (0,112,163),  # false positive # '998ec4' purple
        #    'mask': (247, 200, 200)  # mask color (not used in MCC calculation)
        #    }

        colordict = {
                     'TP': (0,0,0),
                     'TN': (226,226,226),
                     'FN': (201,152,152),
                     'FP': (30,69,230),
                     'mask': (209,209,209)
                     }
    #TODO: else check if mask is specified and add it as color of TN otherwise
    if truth is None:
        truth = np.zeros_like(test)

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
    true_pos = (test==truth & truth)
    true_neg = (test==truth & ~truth)
    false_neg = (truth & ~test)
    false_pos = (test & ~truth)

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

def binary_counts(test, truth, bg_mask=None, score_bg=False):
    """returns TP,TN,FP,FN"""

    true_pos = ((test == truth) & truth)
    true_neg = ((test == truth) & ~truth)
    false_neg = (truth & ~test)
    false_pos = (test & ~truth)

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

    return TP, TN, FP, FN

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


def mcc(test, truth=None, bg_mask=None, score_bg=False, return_counts=False):
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
    if truth is None:
        truth = np.zeros_like(test)

    TP, TN, FP, FN = binary_counts(test, truth, bg_mask=bg_mask,
                                   score_bg=score_bg)


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

def chain_lengths(iterable):

    pos, s = 0, 0

    for b, g in itertools.groupby(iterable):

        if not b:
            # alternative if the bottom doesn't work or something
            #d = deque(enumerate(g,1), maxlen=1)
            #pos += d[0][0] if d else 0

            pos += sum((1 for i in g if not i))

        else:

            s = sum(g)

            yield pos, s

            pos += s

    if not s:
        # so it will return something even if iterable is empty
        yield 0, 0


def _longest_chain_1d(iterable):
    """ will return a tuple of ind, length
    where ind is the position in the iterable the chain starts and length is the
    length of the chain
    """
    return max(chain_lengths(iterable), key=lambda x: x[1])


def longest_chain(arr, axis):
    """Find where the longest chain of boolean values and occurs across an array
    and also return its length
    """

    C = np.apply_along_axis(_longest_chain_1d, axis, arr.astype('bool'))

    start_inds, chain_lens =  np.split(C, 2, axis)

    return np.squeeze(start_inds), np.squeeze(chain_lens)


def _bunch_hists(H, bunches):

    return np.stack((np.sum(np.atleast_2d(H[b,:]), axis=0) for b in bunches))


def scale_to_width_plots(multiscale_approx, max_labels, widths, scales,
                         bunches=None, cmap=None, approx_method=None,
                         figsize=(13,14), style='seaborn', bunch_until=None):
    """
    multiscale_approx is a 3d boolean array whose first dimension is scale
    max_labels is a 2d array of integers that say where the max value of
    F occured. you can get max_labels by running V.argmax(axis=0)

    in widths, each pixel has a unique width

    bunches.flatten() should be the same as arange(scales)
    but can be something like
    ( (0,1,2,3), (4,5), 6, 7, 8, (9,10,11) )

    or even

    (2,3,4,5,(0,1,6,7))

    this is to prevent similar scales from clogging, you can just bin them
    all together.

    approx method is a label to use in the fig titles
    """

    if bunches is None:
        if bunch_until is not None:
            indices = list(range(len(scales)))
            bunches = [indices[:bunch_until],] + indices[bunch_until:]

    plt.style.use(style)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    A = multiscale_approx  # easier to work with

    wbins = np.arange(3,20,2)  # bins of widths in ground truth

    max_hists = [[np.sum((max_labels == s) & (widths==w)) for w in wbins]
                  for s in range(len(scales))]

    hists = np.array([[np.sum((widths==w) & A[n]) for w in wbins]
                      for n in range(len(scales))]
                     )


    if cmap is None:
        # this will just use the default cycle of colors
        colors = np.repeat(None, len(scales))
    else:
        if not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            # try this
            cmap = plt.get_cmap(cmap)

        colors = cmap(np.linspace(0,1,len(scales)))

    labels = [rf'$\sigma_{{{k}}}={sigma:.2f}$'
              for k, sigma in enumerate(scales,1)]

    # number of true positives, false negatives for each width
    tp_hists = [np.sum((widths==w) & A.any(axis=0)) for w in wbins]
    fn_hists = [np.sum((widths==w) & ~A.any(axis=0)) for w in wbins]

    if bunches is not None:
        hists = _bunch_hists(hists, bunches)

        # just return \sigma_{1,2,3} or something rather than listing
        bunch_label = lambda b: r"$\sigma_{{{}}}$".format(','.join(( str(x+1)
                                                                    for x in b)
                                                                   ))

        labels = [labels[b] if np.isscalar(b) else bunch_label(b)
                  for b in bunches]

        if cmap is None:
            # just make it the appropriate length
            cmap = np.repeat(None, len(bunches))
        else:
            colors = [colors[b] if np.isscalar(b) else colors[b[0]]
                      for b in bunches]

    ax[0].bar(wbins, tp_hists, color=(0.6,0.6,0.6),
            label='# true positives')
    ax[0].bar(wbins, fn_hists, bottom=tp_hists, color=(1,.8,.8),
            label='# false negatives')

    for h, mh, label, color in zip(hists, max_hists, labels, colors):
        ax[0].plot(wbins, h, label=label, color=color)
        ax[1].plot(wbins, mh, label=label, color=color)


    ax[0].set_xticks(wbins)
    ax[0].set_xlabel('vessel widths (ground truth), pixels')
    ax[0].set_ylabel('# pixels')
    ax[0].set_xlim(2,21)

    ax[1].set_xticks(wbins)
    ax[1].set_xlabel('vessel widths (ground truth), pixels')
    ax[1].set_ylabel('# pixels')
    ax[1].set_xlim(2,21)

    title = 'pixels reported per scale'
    max_title = r'pixel widths of true positives by scale of $V_\max$'
    if approx_method is not None:
        title += f'({approx_method})'
        max_title += f'({approx_method})'

    ax[0].set_title(title)
    ax[1].set_title(max_title)
    ax[0].legend(loc='best', labelspacing=0.2)
    ax[1].legend(loc='best', labelspacing=0.2)

    fig.tight_layout()
    return fig, ax


def scale_to_argmax_plot(max_labels, widths, scales, normalize=False,
                         bunches=None, cmap=None, figsize=(13,10),
                         style='seaborn-paper', bunch_until=None):
    """
    if normalize, normalize each scale over columns (i.e. all widths)
    multiscale_approx is a 3d boolean array whose first dimension is scale
    max_labels is a 2d array of integers that say where the max value of
    F occured. you can get max_labels by running V.argmax(axis=0)

    in widths, each pixel has a unique width

    bunches.flatten() should be the same as arange(scales)
    but can be something like
    ( (0,1,2,3), (4,5), 6, 7, 8, (9,10,11) )

    or even

    (2,3,4,5,(0,1,6,7))

    this is to prevent similar scales from clogging, you can just bin them
    all together.

    approx method is a label to use in the fig titles
    """

    if bunches is None:
        if bunch_until is not None:
            indices = list(range(len(scales)))
            bunches = [indices[:bunch_until],] + indices[bunch_until:]

    plt.style.use(style)

    fig, ax = plt.subplots(figsize=figsize)

    wbins = np.arange(3,20,2)  # bins of widths in ground truth

    max_hists = np.array([[np.sum((max_labels == s) & (widths==w)) for w in wbins]
                  for s in range(len(scales))])
    if normalize:
        max_hists = max_hists / max_hists.sum(axis=1, keepdims=True)

    if cmap is None:
        # this will just use the default cycle of colors
        colors = np.repeat(None, len(scales))
    else:
        if not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            # try this
            cmap = plt.get_cmap(cmap)

        colors = cmap(np.linspace(0,1,len(scales)))

    labels = [rf'$\sigma_{{{k}}}={sigma:.2f}$'
              for k, sigma in enumerate(scales,1)]

    # number of true positives, false negatives for each width

    if bunches is not None:

        # just return \sigma_{1,2,3} or something rather than listing
        bunch_label = lambda b: r"$\sigma_{{{}}}$".format(','.join(( str(x+1)
                                                                    for x in b)
                                                                   ))

        labels = [labels[b] if np.isscalar(b) else bunch_label(b)
                  for b in bunches]

        if cmap is None:
            # just make it the appropriate length
            cmap = np.repeat(None, len(bunches))
        else:
            colors = [colors[b] if np.isscalar(b) else colors[b[0]]
                      for b in bunches]

    #ax.bar(wbins, tp_hists, color=(0.6,0.6,0.6),
    #        label='# true positives')
    #ax.bar(wbins, fn_hists, bottom=tp_hists, color=(1,.8,.8),
    #        label='# false negatives')

    for mh, label, color in zip(max_hists, labels, colors):
        ax.plot(wbins, mh, label=label, color=color)



    ax.set_xticks(wbins)
    ax.set_xlabel('vessel widths (ground truth), pixels')
    if normalize:
        ax.set_ylabel('# pixels identified by scale /'
                      '# pixels identified by all scales')
    else:
        ax.set_ylabel('# pixels')

    ax.set_xlim(2,21)

    max_title = r'pixel widths of true positives by scale of $V_\max$'

    ax.set_title(max_title)
    ax.legend(loc='best', labelspacing=0.2)

    fig.tight_layout()
    return fig, ax


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


