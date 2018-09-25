#!/usr/bin/env python3

import numpy as np
from skimage.data import camera
from skimage.io import imread

import matplotlib.pyplot as plt
from hfft import gauss_freq, blur, fft_gaussian, fft_hessian
from scipy.ndimage import gaussian_filter

from scipy.linalg import norm
import timeit

#img = camera() / 255.
img = imread('samples/barium1.png', as_grey=True) / 255.
mask = imread('samples/barium1.mask.png', as_grey=True)

# compare computation speed over sigmas

# N logarithmically spaced scales between 1 and 2^m
N = 5
m = 8
sigmas = np.logspace(0,m, num=N, base=2)

fft_results = list()
std_results = list()

for sigma in sigmas:
    # test statements to compare (fft-based gaussian vs convolution-based)
    fft_test_statement = 'fft_gaussian(img,{})'.format(sigma)
    std_test_statement = 'gaussian_filter(img,{})'.format(sigma)
    # run each statement 1 times (with 2 runs in each trial)
    # returns/appends the average of 3 runs
    fft_results.append(timeit.timeit(fft_test_statement,
                                 number=1, globals=globals()))
    std_results.append(timeit.timeit(std_test_statement,
                                 number=1, globals=globals()))

    # now actually evaluate both to compare
    f = eval(fft_test_statement)
    s = eval(std_test_statement)

    # normalize each matrix by frobenius norm and take difference
    # ideally should try to zero out the "mask" area
    diff = np.abs(f / norm(f) - s / norm(s))
    raw_diff = np.abs(f - s)
    # don't care if it's the background
    diff[mask==1] = 0
    raw_diff[mask==1] = 0

    # should format this stuff better into a legible table
    print(sigma, diff.max(), raw_diff.max())

lines = plt.plot(sigmas, fft_results, 'go', sigmas, std_results, 'bo')
plt.xlabel('sigma (gaussian blur parameter)')
plt.ylabel('run time (seconds)')
plt.legend(lines, ('fft-gaussian', 'conv-gaussian'))
plt.title('Comparision of Gaussian Blur Implementations')
