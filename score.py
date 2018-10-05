#!/usr/bin/env python3

import numpy as np
from get_placenta import open_typefile, open_tracefile

def confusion(a, b, a_color=None, b_color=None):
    """
    create graphical confusion matrix for 2D boolean arrays of the
    same size.

    INPUTS:
        a - first matrix a 2D boolean matrix
        b - second matrix, a 2D boolean matrix of the same size as a
        a_color : which color to use (default is a nice blue)
        b_color : which color to use (default is inverse color of a)

    OUTPUT:
        a matrix of size a, where
            pixels in a exclusively a are a_color
            pixels in a exclusively b are b_color
            pixels in both are white (unless b_color has been set)
            pixels in neither are black

    """

def confusion_4(test, truth):
    """
    distinct coloration of false positives and negatives.
    supply HEX values

    colors output matrix with
    true_pos if test[-] == truth[-] == 1
    true_neg if test[-] == truth[-] == 0
    false_neg if test[-] == 0 and truth[-] == 1
    false_pos if test[-] == 1 and truth[-] == 0
    """
    true_neg_color = np.array([247,247,247], dtype='f')/255 # 'f7f7f7'
    true_pos_color = np.array([0, 0, 0] , dtype='f')/255  # '000000'
    false_neg_color = np.array([241,163,64], dtype='f')/255# 'f1a340' orange
    false_pos_color = np.array([153,142,195], dtype='f')/255 # '998ec4' purple

    #    if a_color is None:
    #        a_color = np.array([0.9, 0.6, 0.1])
    #    if b_color is None:
    #        b_color = 1 - a_color
    #

    #a_c = np.tile(a[:,:,np.newaxis], (1, 1, 3)) * a_color
    #b_c = np.tile(b[:,:,np.newaxis], (1, 1, 3)) * b_color

    #return a_c + b_c

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


def mcc(test, truth, bg_mask=None, return_counts=False):
    """
    Matthews correlation coefficient
    returns a float between -1 and 1
    -1 is total disagreement between test & truth
    0 is "no better than random guessing"
    1 is perfect prediction

    """
    true_pos = np.bitwise_and(test==truth, truth)
    true_neg = np.bitwise_and(test==truth, np.invert(truth))
    false_neg = np.bitwise_and(truth, np.invert(test))
    false_pos = np.bitwise_and(test, np.invert(truth))

    if bg_mask is not None:
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

    C = confusion_4(A,B)

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


