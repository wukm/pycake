#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
from scipy import signal
import scipy.fftpack as fftpack
from scipy.special import iv, ive
from scipy.ndimage import gaussian_filter
from itertools import combinations_with_replacement

from numpy.linalg import eigvals

# for demos
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.util import img_as_float
from skimage.measure import compare_mse, compare_nrmse

"""
hfft.py is the implementation of calculating the hessian of a real image based
in frequency space (rather than direct convolution with a gaussian, as is
standard in scipy, for example). This file also contains implementations of the
discrete gaussian kernel, and a visual demonstration of the "semigroup
property" of each implementation.


TODO: PROVIDE MAIN USAGE NOTES
"""

def fft_gaussian(img,sigma, kernel=None):

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
    # output of signal.gaussian is normalized to 1 so you need to scale
    # it back to work
    #A = 1 / (2*np.pi*sigma**2) # scale factor for 2D

    if kernel in ('discrete', None):
        kern_x = discrete_gaussian_kernel(img.shape[0], sigma)
        kern_y = discrete_gaussian_kernel(img.shape[1], sigma)
    elif kernel == 'sampled':
        # scaling factor for 1d gaussian (use it twice)
        A = 1 / (np.sqrt((2*np.pi)*sigma**2))
        kern_x = A*signal.gaussian(img.shape[0], sigma)
        kern_y = A*signal.gaussian(img.shape[1], sigma)
    else:
        raise ValueError("Key must be 'discrete' or 'sampled'")

    kernel = np.outer(kern_x, kern_y)

    return signal.fftconvolve(img, kernel, mode='same')


def discrete_gaussian_kernel(n_samples, sigma):
    """
    sigma is the scale, n_samples is the number of samples to compute
    will return a window centered a zero
    i.e. arange(-n_samples//2, n_samples//2+1)

    not sure how to center it on zero though, since integers only

    note! to make this work similarly to fft_gaussian, this uses
    sigma = np.sqrt(t). Usually you'll find this in terms of t

    by using scipy.special.ive instead we prevent blowups
    """
    dom = np.arange(-(n_samples//2), (n_samples//2) + 1)
    #there should be a scaling parameter alpha but whatever
    #return np.exp(-t) * iv(dom,t)
    if (n_samples % 2) == 0:
        dom = dom[1:]
    return ive(dom,sigma**2)


def fft_dgk(img,sigma,order=0,A=None):
    """
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


def fft_weingarten_eigs(img, sigma=1., kernel=None):
    """
    Compute eigenvalues actual weingarten map. This isn't general use,
    works only with grayscale masked arrays
    """

    gaussian_filtered = fft_gaussian(img, sigma=sigma, kernel=kernel)

    gradients = np.gradient(gaussian_filtered)

    hx, hy = gradients

    axes = range(img.ndim)

    hxx, hxy, hyy = [np.gradient(gradients[ax0], axis=ax1)
                    for ax0, ax1 in combinations_with_replacement(axes, 2)]


    # need to do gf*(G@H)
    gf = (1 + hx**2 + hy**2)**(-1.5)


    # each of these will have shape (2,2, *img.shape)
    G = np.array([[ 1 + hy**2 , -hx*hy],
                  [ -hx*hy, 1 + hx**2]])
    H = np.array([[hxx,hxy],
                 [hxy, hyy]])

    # where to store eigenvalues
    K2 = np.zeros_like(img, dtype='float64')
    K1 = np.zeros_like(img, dtype='float64')


    ## build the 2x2 weingarten map at each pixel and find its eigvals
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img.mask[i,j]:
                continue # don't bother with things outside the masked region
            L = eigvals(gf[i,j]*(H[...,i,j] @ G[...,i,j]))

            K1[i,j], K2[i,j] = L

    """
    # we hand code it individually, knowing that the eigs are
    # T/2 +- sqrt(T^2/4 - D)
    # where T is the trace and D is the determinant
    # the determinant of HG is det(H)det(G)
    # and trace is easy
    # this could be optimized by eliminating repeated calculations
    #T = hxx*hyy + hxx*hy*hy + hyy*hx*hx - 2*hxy*hx*hy
    #D = (1+hx*hx)*(1+hy*hy) - hx*hx*hy*hy
    #D *= (hxx*hyy - hxy*hxy)
    gd = 1 + hy**2
    ge = -hx*hy
    gf = 1 + hx**2

    # this is the trace divided by 2
    T = (hxx*gd + hyy*gf + 2*hxy*ge)/2
    D = (hxx*hyy - hxy**2) * (gd*gf - ge**2)

    T = ma.masked_array(T, mask=img.mask)
    D = ma.masked_array(D, mask=img.mask)


    discriminant = T**2 - D
    #plt.imshow((discriminant<0))
    #plt.title((discriminant<0).sum())
    #plt.show()
    discriminant = np.sqrt(discriminant.filled(0))
    K1 = T + discriminant
    K2 = T - discriminant

    K1 *= gf
    K2 *= gf
    """
    # same thing as reorder_eigs
    L = np.stack((K1,K2))
    mag = np.argsort(np.abs(L), axis=0)
    K1, K2 = np.take_along_axis(L, mag, axis=0)

    return K1, K2


def fft_hessian(image, sigma=1., kernel=None):
    """
    a reworking of skimage.feature.hessian_matrix that uses
    e FFT to compute gaussian, which results in a considerable speedup

    INPUT:
        image - a 2D image (which type?)
        sigma - coefficient for gaussian blur
        kernel - input to fft_gaussian
        gradient - if you've already computed this

    OUTPUT:
        (Lxx, Lxy, Lyy) - a triple containing three arrays
            each of size image.shape containing the xx, xy, yy derivatives
            respectively at each pixel. That is, for the pixel value given
            by image[j][k] has a calculated 2x2 hessian of
            [ [Lxx[j][k], Lxy[j][k]],
              [Lxy[j][k], Lyy[j][k]] ]
    """

    gaussian_filtered = fft_gaussian(image, sigma=sigma, kernel=kernel)

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


def demo(img=None):
    """
    old main function for testing.

    This simply tests fft_gaussian on a test image,
    """

    if img is None:
        img = img_as_float(camera())
    else:
        img = img_as_float(img)

    sample_sigmas = (.5, 2, 8, 30)
    #sample_sigmas = (.2, 2)

    # build the graphs here side by side
    # show regular blur, sampled blur, discrete blur, 1d plot of signals
    # so a 4 by 4 grid

    fig, axes = plt.subplots(nrows=len(sample_sigmas), ncols=4,
                             figsize=(10, 10))

    for cax, sigma in enumerate(sample_sigmas):

        # convolve the image with a gaussian kernel, one of three ways
        fft_dgk = fft_gaussian(img, sigma, kernel='discrete')
        fft_sampled = fft_gaussian(img, sigma, kernel='sampled')
        xy_sampled = gaussian_filter(img, sigma, mode='constant', cval=0)

        # make the fancy sample
        N = 80
        dom = np.arange(-(N//2), N//2 + 1)
        dgk = discrete_gaussian_kernel(N,  sigma)
        A = np.sqrt(2*np.pi*sigma**2)
        A = 1 / A
        sgk = A * signal.gaussian(N+1, sigma)

        axes[cax, 0].imshow(xy_sampled, cmap='gray', vmin=0, vmax=1)
        axes[cax, 0].set_ylabel(r'$\sigma={}$'.format(sigma))
        #axes[cax, 0].set_title(f'ndi.gaussian_filter, σ={sigma}')
        #axes[cax, 0].axis('off')
        axes[cax, 0].set_xticks([])
        axes[cax, 0].set_yticks([])

        axes[cax, 1].imshow(fft_sampled, cmap='gray', vmin=0, vmax=1)
        #axes[cax, 1].imshow(fft_sampled, cmap='gray')
        #axes[cax, 1].set_title(f'fft sampled kernel, σ={sigma}')
        axes[cax, 1].axis('off')

        axes[cax, 2].imshow(fft_dgk, cmap='gray', vmin=0, vmax=1)
        #axes[cax, 2].set_title(f'fft discrete kernel, σ={sigma}')
        axes[cax, 2].axis('off')


        axes[cax, 3].plot(dom, sgk, 'k', dom, dgk, 'g:')
        #axes[cax, 3].set_title(f'discrete vs. sampled kernel σ={sigma}')
        #axes[cax, 3].axes.set_aspect('equal')

    # set titles for the first column
    axes[0,0].set_title('(a)')
    axes[0,1].set_title('(b)')
    axes[0,2].set_title('(c)')
    axes[0,3].set_title('(d)')

    plt.tight_layout()
    plt.show()


def compare_mae(arr1, arr2):

    assert arr1.shape == arr2.shape
    return np.abs(arr1 - arr2).sum() / arr1.size


def semigroup_demo(img=None):
    """
    the step ones don't look anywhere near as blurred as the initial image
    in any case! don't use this till it's good!
    """
    if img is None:
        img = img_as_float(camera())
    else:
        img = img_as_float(img)

    sigma = 45.
    n_steps = 2
    sigmas = (10,35)

    fft_discrete = fft_gaussian(img, sigma, kernel='discrete')
    fft_sampled = fft_gaussian(img, sigma, kernel='sampled')
    xy_sampled = gaussian_filter(img, sigma, mode='constant', cval=0)

    step_discrete = img.copy()
    step_fft_sampled = img.copy()
    step_xy_sampled = img.copy()

    #sigma_n = sigma / n_steps
    #sigma_n = np.power(sigma, 1/n_steps)
    counter = 0
    for sigma_n in sigmas:
        step_discrete = fft_gaussian(step_discrete, sigma_n, kernel='discrete')
        step_fft_sampled = fft_gaussian(step_fft_sampled, sigma_n,
                                        kernel='sampled')
        step_xy_sampled = gaussian_filter(step_xy_sampled, sigma_n,
                                          mode='constant', cval=0)
        counter += sigma_n
        #print(counter, end=' ')

    print()
    er = int(sigma)
    crop = np.s_[er:-er,er:-er]

    fft_discrete = fft_discrete[crop]
    fft_sampled = fft_sampled[crop]
    xy_sampled = xy_sampled[crop]
    step_discrete = step_discrete[crop]
    step_fft_sampled = step_fft_sampled[crop]
    step_xy_sampled = step_xy_sampled[crop]

    fig, axes = plt.subplots(ncols=3, nrows=2)
    axes[0,0].imshow(xy_sampled, vmin=0, vmax=1, cmap='gray')
    axes[0,0].set_title('(a)')
    axes[0,0].set_xticks([]), axes[0,0].set_yticks([])

    axes[0,1].imshow(fft_sampled, vmin=0, vmax=1, cmap='gray')
    axes[0,1].set_title('(b)')
    axes[0,1].set_xticks([]), axes[0,1].set_yticks([])

    axes[0,2].imshow(fft_discrete, vmin=0, vmax=1, cmap='gray')
    axes[0,2].set_title('(c)')
    axes[0,2].set_xticks([]), axes[0,2].set_yticks([])

    axes[1,0].imshow(step_xy_sampled, vmin=0, vmax=1, cmap='gray')
    axes[1,0].axis('off')
    axes[1,1].imshow(step_fft_sampled, vmin=0, vmax=1, cmap='gray')
    axes[1,1].axis('off')
    axes[1,2].imshow(step_discrete, vmin=0, vmax=1, cmap='gray')
    axes[1,2].axis('off')

    MSE_sampled = compare_mse(xy_sampled, step_xy_sampled)
    MSE_fft_sampled = compare_mse(fft_sampled, step_fft_sampled)
    MSE_discrete = compare_mse(fft_discrete, step_discrete)

    print(f'MSE sampled:{MSE_sampled}')
    print(f'MSE fft_sampled:{MSE_fft_sampled}')
    print(f'MSE discrete:{MSE_discrete}')
    print()
    #NRMSE_sampled = compare_nrmse(xy_sampled, step_xy_sampled)
    #NRMSE_fft_sampled = compare_nrmse(fft_sampled, step_fft_sampled)
    #NRMSE_discrete = compare_nrmse(fft_discrete, step_discrete)

    #print(f'NRMSE sampled:{NRMSE_sampled}')
    #print(f'MSE fft_sampled:{NRMSE_fft_sampled}')
    #print(f'NRMSE discrete:{NRMSE_discrete}')
    #for ax, title in zip(axes.ravel(), ['(a)', '(b)', '(c)', '(d)']):
    #    ax.axis('off')
    #    ax.set_title(title)


    MAE_sampled = compare_mae(xy_sampled, step_xy_sampled)
    MAE_fft_sampled = compare_mae(fft_sampled, step_fft_sampled)
    MAE_discrete = compare_mae(fft_discrete, step_discrete)

    print(f'MAE sampled:{MAE_sampled}')
    print(f'MAE fft_sampled:{MAE_fft_sampled}')
    print(f'MAE discrete:{MAE_discrete}')

    plt.show()

if __name__ == "__main__":
    from skimage.io import imread
    from placenta import list_by_quality, get_named_placenta
    #A = list_by_quality(0)[0]
    #A = get_named_placenta(A)
    #A = imread('samples/5.3.02.tiff')
    A = None
    demo(A)
    semigroup_demo(A)
