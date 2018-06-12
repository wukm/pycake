"""

here you want to show  the accuracy of hfft.py

BOILERPLATE

show that gaussian blur of hfft is accurate, except around the boundary
proportional to sigma.

or if they're off by a scaling factor, show that the derivates (taken the same way)
are proportional.

pseudocode

A = gaussian_blur(image, sigma, method='convential')
B = gaussian_blue(image, sigma, method='fourier')

zero_order_accurate = isclose(A, B, tol)

J_A= get_jacobian(A)
J_B = get_jacobian(B)

first_order_accurate = isclose(J_A, J_B, tol)

A_eroded = zero_around_plate(A, sigma)
B_eroded = zero_around_plate(B, sigma)

J_A_eroded = zero_around_plate(A, sigma)
J_B_eroded = zero_around_plate(B, sigma)

zero_order_accurate_no_boundary = isclose(A_eroded, B_eroded, tol)
first_order_accurate = isclose(J_A_eroded, J_B_eroded, tol)

"""

from get_placenta import get_named_placenta

from hfft import fft_hessian, fft_gaussian
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from get_placenta import mimshow

import numpy as np
from scipy.ndimage import laplace
import numpy.ma as ma

imgfile = 'barium1.png'
maskfile = 'barium1.mask.png'

img_raw = get_named_placenta(imgfile, maskfile=maskfile)

# so that scipy.ndimage.gaussian_filter doesn't use uint8 precision (jesus)
img = img_raw / 255.

# show a matrix with img.mask mask
ms = lambda x: mimshow(ma.masked_array(x, img.mask))

sigma = 2

print('applying standard gauss blur')
# THIS USES THE SAME DTYPE AS THE INPUT SO DEAR LORD MAKE SURE IT'S A FLOAT
A = gaussian_filter(img.astype('f'), sigma, mode='constant') #zero padding
print('applying fft gauss blur')
B = fft_gaussian(img, sigma)

print('calculating first derivatives')
Ax, Ay = np.gradient(A)
Bx, By = np.gradient(B)


print('calculating second derivatives')
Axx, Axy = np.gradient(Ax)
Ayx, Ayy = np.gradient(Ay)

Bxx, Bxy = np.gradient(Bx)
Byx, Ayy = np.gradient(By)

print('done.')

# note: A & B are off but have the same shape
# rescale to [0,255] (actually should keep as 0,1? )
A_unscaled = A.copy()
B_unscaled = B.copy()

A = 255*(A-A.min())/(A.max()-A.min())
B = 255*(B-B.min())/(B.max()-B.min())

# the following shows a random vertical slice of A & B (when scaled)
# the results are even more fitting when you scale B to coincide with A's max
# (which obviously isn't feasible in practice)

# FIXEDISH AFTER SCALING!

plt.plot(np.arange(A.shape[1]),A[A.shape[0]//2,:],
         label='scipy.ndimage,gaussian_filter')
plt.plot(np.arange(B.shape[1]), B[B.shape[0]//2,:],
         label='fft_gaussian')
plt.legend()
