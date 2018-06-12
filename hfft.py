#!/usr/bin/env python3

import numpy as np
from scipy import signal
import scipy.fftpack as fftpack

"""
hfft.py is the implementation of calculating the hessian of a real

image based in frequency space (rather than direct convolution with a gaussian
as is standard in scipy, for example).

TODO: PROVIDE MAIN USAGE NOTES
"""

def gauss_freq(shape, σ=1.):
    """
    DEPRECATED

    NOTE:
        this function is/should be? for illustrative purposes only--
        we can actually build this much faster using the builtin
        scipy.signal.gaussian rather than a roll-your-own

    build a shape=(M,N) sized gaussian kernel in frequency space
    with size σ

    (due to the convolution theorem for fourier transforms, the function
    created here may simply be *multiplied" against the signal.

    """

    M,  N = shape
    fgauss = np.fromfunction(lambda μ,ν: ((μ+M+1)/2)**2 + ((ν+N+1)/2)**2, shape=shape)

    # is this used?
    coeff = (1 / (2*np.pi * σ**2))

    return np.exp(-fgauss / (2*σ**2))

def blur(img, sigma):
    """
    DEPRECATED
    a roll-your-own FFT-implemented gaussian blur.
    fftgauss below is preferred (it is more efficient)
    """

    I = fftpack.fft2(img) # get 2D transform of the image

    # do whatever

    I *= gauss_freq(I.shape, sigma)


    return fftpack.ifft2(I).real

def fftgauss(img,sigma):

    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html

    in particular the example in which a gaussian blur is implemented.

    along with the comment:
    "Gaussian blur implemented using FFT convolution. Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries. The
    convolve2d function allows for other types of image boundaries, but is far
    slower"

    (i.e. doesn't use FFT).

    note that here, you actually take the FFT of a gaussian (rather than
    build it in frequency space). there are ~6 ways to do this.
    """

    #create a 2D gaussian kernel to take the FFT of
    kernel = np.outer(signal.gaussian(img.shape[0], sigma),
                        signal.gaussian(img.shape[1],sigma))

    return signal.fftconvolve(img, kernel, mode='same')

def fft_hessian(image, sigma=1):
    """
    a reworking of skimage.feature.hessian_matrix that uses
    e FFT to compute gaussian, which results in a considerable speedup

    INPUT:
        image - a 2D image (which type?)
        sigma - coefficient for gaussian blur

    OUTPUT:
        (Lxx, Lxy, Lyy) - a triple containing three arrays
            each of size image.shape containing the xx, xy, yy derivatives
            respectively at each pixel. That is, for the pixel value given
            by image[j][k] has a calculated 2x2 hessian of
            [ [Lxx[j][k], Lxy[j][k]],
              [Lxy[j][k], Lyy[j][k]] ]
    """

    gaussian_filtered = fftgauss(image, sigma=sigma)

    Lx, Ly = np.gradient(gaussian_filtered)

    Lxx, Lxy = np.gradient(Lx)
    Lxy, Lyy = np.gradient(Ly)

    return (Lxx, Lxy, Lyy)


def _old_test():
    """
    old main function for testing.

    This simply tests fftgauss on a test image, exemplifying the speedup
    compared to a traditional gaussian.
    """
    import matplotlib.pyplot as plt

    from skimage.data import camera

    img = camera() / 255.

    sample_sigmas = (.2, 2, 10, 30)

    outputs = (fftgauss(img, sample_sigmas[0]),
               fftgauss(img, sample_sigmas[1]),
               fftgauss(img, sample_sigmas[2]),
               fftgauss(img, sample_sigmas[3]),
               )


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(outputs[0], cmap='gray')
    axes[0, 0].set_title('fftgauss σ={}'.format(sample_sigmas[0]))
    axes[0, 0].axis('off')

    axes[0, 1].imshow(outputs[1], cmap='gray')
    axes[0, 1].set_title('fftgauss σ={}'.format(sample_sigmas[1]))
    axes[0, 1].axis('off')

    axes[1, 0].imshow(outputs[2], cmap='gray')
    axes[1, 0].set_title('fftgauss σ={}'.format(sample_sigmas[2]))
    axes[1, 0].axis('off')

    axes[1, 1].imshow(outputs[3], cmap='gray')
    axes[1, 1].set_title('fftgauss σ={}'.format(sample_sigmas[3]))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":

    pass
