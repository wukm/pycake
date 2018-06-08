#!/usr/bin/env python3

import numpy as np

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
