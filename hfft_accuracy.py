"""

here you want to show  the accuracy of hfft.py

BOILERPLATE

show that gaussian blur of hfft is accurate, except potentially around the
boundary proportional to sigma.

or if they're off by a scaling factor, show that the derivates
(taken the same way) are proportional.

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

from score import mean_squared_error
import numpy as np
from scipy.ndimage import laplace
import numpy.ma as ma

from skimage.segmentation import find_boundaries
from skimage.morphology import disk, binary_dilation

from diffgeo import principal_curvatures
from frangi import structureness, anisotropy

def erode_plate(img, sigma, mask=None):
    """
    Apply an eroded mask to an image
    assume (if helpful) that the boundary of the placenta is a connected loop
    that is, there is a single inside and outside of the shape, and that
    the placenta is more or less convex
    """

    if mask is None:
        mask = img.mask

    # get a boolean array that is 1 along the border of the mask, zero elsewhere
    # default mode is 'thick' which is fine
    bounds = find_boundaries(mask)

    # structure element to dilate by is a disk of diameter sigma
    # rounded up to the nearest integer. this may be too conservative.
    selem = disk(np.ceil(sigma))
    dilated_border = binary_dilation(bounds, selem=selem)

    new_mask = np.logical_or(mask, dilated_border)

    return ma.masked_array(img, mask=new_mask)

# FIX SOME ISSUES, BINARY DILATION IS TAKING HELLA LONG AND ALSO
# THERE ARE RANDOM BLIPS INSIDE THE MASK!!!
# FIX IN GIMP!:

imgfile = 'barium1.png'
maskfile = 'barium1.mask.png'

img_raw = get_named_placenta(imgfile, maskfile=maskfile)

# so that scipy.ndimage.gaussian_filter doesn't use uint8 precision (jesus)
img = img_raw / 255.

# convenience function to show a matrix with img.mask mask
ms = lambda x: mimshow(ma.masked_array(x, img.mask))

sigma = 11.

print('applying standard gauss blur')
# THIS USES THE SAME DTYPE AS THE INPUT SO DEAR LORD MAKE SURE IT'S A FLOAT
A = gaussian_filter(img.astype('f'), sigma, mode='constant') #zero padding
print('applying fft gauss blur')
B = fft_gaussian(img, sigma)
B_unnormalized = B.copy()
B = B / (2*(sigma**2)*np.pi)

A = erode_plate(A, sigma, mask=img.mask)
B = erode_plate(B, sigma, mask=img.mask)
print('calculating first derivatives')
Ax, Ay = np.gradient(A.filled(0))
Bx, By = np.gradient(B.filled(0))


print('calculating second derivatives')

# you can verify np.isclose(Axy,Ayx) && np.isclose(Bxy,Byx) -> True
Axx, Axy = np.gradient(Ax)
Ayx, Ayy = np.gradient(Ay)

Bxx, Bxy = np.gradient(Bx)
Byx, Byy = np.gradient(By)



print('calculating eigenvalues of hessian')
ak1, ak2 = principal_curvatures(A, sigma=sigma, H=(Axx,Axy,Ayy))
bk1, bk2 = principal_curvatures(B, sigma=sigma, H=(Bxx,Bxy,Byy))

R1 = anisotropy(ak1,ak2)
R2 = anisotropy(bk1,bk2)

S1 = structureness(ak1, ak2)
S2 = structureness(bk1, bk2)
print('done.')

# even without scaling (which occurs below) the second derivates should be
# close. normalize matrices using frobenius norm of the hessian?
# note: A & B are off but have the same shape


# rescale to [0,255] (actually should keep as 0,1? )
#A_unscaled = A.copy()
#B_unscaled = B.copy()

#Ascaled = (A-A.min())/(A.max()-A.min())
#Bscaled = (B-B.min())/(B.max()-B.min())

# the following shows a random vertical slice of A & B (when scaled)
# the results are even more fitting when you scale B to coincide with A's max
# (which obviously isn't feasible in practice)

# FIXEDISH AFTER SCALING!

plt.plot(np.arange(A.shape[1]),A[A.shape[0]//2,:],
         label='scipy.ndimage,gaussian_filter')
plt.plot(np.arange(B.shape[1]), B[B.shape[0]//2,:],
         label='fft_gaussian')
plt.legend()

#MSE = ((A-B)**2).sum() / A.size
MSE = mean_squared_error(A,B)
