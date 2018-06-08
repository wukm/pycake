#!/usr/bin/env python3

import numpy as np
from skimage.data import camera
from skimage.io import imread

import matplotlib.pyplot as plt
from hfft import gauss_freq, blur, fftgauss, fft_hessian
from scipy.ndimage import gaussian_filter

from scipy.linalg import norm
import timeit

#img = camera() / 255.
img = imread('raw_barium.png') / 255.

# compare computation speed over sigmas

# N logarithmically spaced scales between 1 and 2^m
N = 20
m = 8
sigmas = np.logspace(0,m, num=N, base=2)

fft_results = list()
std_results = list()

for sigma in sigmas:
    # test statements to compare (fft-based gaussian vs convolution-based)
    fft_test_statement = 'fftgauss(img,{})'.format(sigma)
    std_test_statement = 'gaussian_filter(img,{})'.format(sigma)
    # run each statement 3 times (with 2 runs in each trial)
    # returns/appends the average of 3 runs
    fft_results.append(timeit.timeit(fft_test_statement,
                                 number=3, globals=globals()))
    std_results.append(timeit.timeit(std_test_statement,
                                 number=3, globals=globals()))

    # now actually evaluate both to compare
    f = eval(fft_test_statement)
    s = eval(std_test_statement)

    # calculate frob norm
    print(sigma, norm(f-s))

lines = plt.plot(sigmas, fft_results, 'go', sigmas, std_results, 'bo')
plt.xlabel('sigma (gaussian blur parameter)')
plt.ylabel('run time (seconds)')
plt.legend(lines, ('fft-gaussian', 'conv-gaussian'))
plt.title('Comparision of Gaussian Blur Implementations')
