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

    if a_color is None:
        a_color = np.array([0.9, 0.6, 0.1])
    if b_color is None:
        b_color = 1 - a_color


    a_c = np.tile(a[:,:,np.newaxis], (1, 1, 3)) * a_color
    b_c = np.tile(b[:,:,np.newaxis], (1, 1, 3)) * b_color

    return a_c + b_c

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from skimage.data import binary_blobs

    A = binary_blobs()
    B = binary_blobs()

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
            trace = open_typefile(filename, 'trace')

    # calculate the confusion matrix
    # assert same size and dimension?
    C = confusion(approx, trace, a_color, b_color)

    return C

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
