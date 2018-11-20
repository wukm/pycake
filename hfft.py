#!/usr/bin/env python3

import numpy as np
from scipy import signal
import scipy.fftpack as fftpack
from scipy.special import iv
from itertools import combinations_with_replacement

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
    fft_gaussian below is preferred (it is more efficient)
    """

    I = fftpack.fft2(img) # get 2D transform of the image

    # do whatever

    I *= gauss_freq(I.shape, sigma)


    return fftpack.ifft2(I).real

def fft_gaussian(img,sigma,A=None):

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

    # scale factor!
    A = 1 / (2*np.pi*sigma**2)
    kernel = np.outer(A*signal.gaussian(img.shape[0], sigma),
                        A*signal.gaussian(img.shape[1],sigma))

    return signal.fftconvolve(img, kernel, mode='same')

def discrete_gaussian_kernel(n_samples, t):
    """
    t is the scale, n_samples is the number of samples to compute
    will return a window centered a zero
    i.e. arange(-n_samples//2, n_samples//2+1)

    note! to make this work similarly to fft_gaussian, you should pass
    sigma**2 into t here. Figure out why?
    """
    dom = np.arange(-n_samples//2, n_samples // 2 + 1)
    #there should be a scaling parameter alpha but whatever
    return np.exp(-t) * iv(dom,t)

def fft_dgk(img,sigma,order=0,A=None):
    """
    A is scaling factor.
    This is the discrete gaussian kernel which is supposedly less crappy
    than using a sampled gaussian.
    """
    m,n = img.shape
    # i don't know if this will suck if there are odd dimensions
    kernel = np.outer(discrete_gaussian_kernel(m,sigma**2),
                      discrete_gaussian_kernel(n,sigma**2))

    return signal.fftconvolve(img, kernel, mode='same')

def fft_fdgk(img,sigma):
    """
    convolve with discrete gaussian kernel in freq. space
    """
    # this would be a lot better since you wouldn't have to deal
    # with an arbitrary cutoff of size of the discrete kernel
    # since the freq. space version is just
    # exp{\alpha*t (cos\theta - 1)}
    # see formula 22 of lindeberg discrete paper

    pass

def fft_hessian(image, sigma=1., kernel=None):
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
    if kernel == 'discrete':
        #print('using discrete kernel!')
        gaussian_filtered = fft_dgk(image, sigma=sigma)
    else:
        #print('using sampled gauss kernel')
        gaussian_filtered = fft_gaussian(image, sigma=sigma)

    gradients = np.gradient(gaussian_filtered)

    axes = range(image.ndim)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]

    return H_elems

def fft_gradient(image, sigma=1.):
    """ returns gradient norm """

    gaussian_filtered = fft_gaussian(image, sigma=sigma)

    Lx, Ly = np.gradient(gaussian_filtered)

    return np.sqrt(Lx**2 + Ly**2)
def _old_test():
    """
    old main function for testing.

    This simply tests fft_gaussian on a test image, exemplifying the speedup
    compared to a traditional gaussian.
    """
    import matplotlib.pyplot as plt

    from skimage.data import camera

    img = camera() / 255.

    sample_sigmas = (.2, 2, 10, 30)

    outputs = (fft_gaussian(img, sample_sigmas[0]),
               fft_gaussian(img, sample_sigmas[1]),
               fft_gaussian(img, sample_sigmas[2]),
               fft_gaussian(img, sample_sigmas[3]),
               )


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(outputs[0], cmap='gray')
    axes[0, 0].set_title('fft_gaussian σ={}'.format(sample_sigmas[0]))
    axes[0, 0].axis('off')

    axes[0, 1].imshow(outputs[1], cmap='gray')
    axes[0, 1].set_title('fft_gaussian σ={}'.format(sample_sigmas[1]))
    axes[0, 1].axis('off')

    axes[1, 0].imshow(outputs[2], cmap='gray')
    axes[1, 0].set_title('fft_gaussian σ={}'.format(sample_sigmas[2]))
    axes[1, 0].axis('off')

    axes[1, 1].imshow(outputs[3], cmap='gray')
    axes[1, 1].set_title('fft_gaussian σ={}'.format(sample_sigmas[3]))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    pass
