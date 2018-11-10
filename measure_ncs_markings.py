#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import imread
from skimage.morphology import disk
CYAN = [0,255,255]
YELLOW = [255,255,0]

def measure_ncs_markings(ucip_img):
    """
    find location of ucip and resolution of image based on input
    (similar to perimeter layer in original NCS data set
    """

    # just in case it's got an alpha channel, remove it
    img = ucip_img[:,:,0:3]

    # given the image img (make sure no alpha channel)
    # find all cyan pixels (there are two boxes of 3 pixels each and we
    # just want to extract the middle of each
    print('the image size is {}x{}'.format(img.shape[0], img.shape[2]))
    rulemarks = np.all(img == CYAN, axis=-1)

    # turn into two pixels (these should each by shape (18,)
    X,Y = np.where(rulemarks)

    assert X.shape == Y.shape
    assert X.size == 18
    # the two pixels at the center of each box
    A, B = (X[4], Y[4]) , (X[13], Y[13])

    ruler_distance =  np.sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 )
    print(f'one cm equals {ruler_distance} pixels')

    # the umbillical cord insertion point (UCIP) is a yellow circle
    # of radius 19
    ucipmarks = np.all(img == YELLOW, axis=-1)
    X,Y = np.where(ucipmarks)

    # find midpoint of the x & y cooridnates

    assert X.max() - X.min() == Y.max() - Y.min()
    radius = (X.max() - X.min()) // 2

    mid = (X.min() + radius, Y.min() + radius)
    print('the middle of the UCIP location is', mid)
    print('the radius outward is', radius)
    print('the total measurable diameter is', radius*2 + 1)

    return mid, ruler_distance

def add_ucip_to_mask(m, radius=100, mask=None, size=None):
    """
    - m is a tuple (2x1) then it's the midpoint of the ucip, add it to the mask
    - if no mask is supplied, dilate the point in an array of zeros size `size`
    (would be the same as passing mask=np.zeros(size))
    - radius is the dilation radius
    """

    if mask is None:
        mask = np.zeros(size)

    # this is way faster than dilating the point in the matrix,
    # just set this at the centered point

    # doesn't check for out of bounds stuff. use at your own peril
    D = disk(radius)
    mask[m[0]-radius:m[0]+radius+1 , m[1]-radius:m[1]+radius+1] = D

    return mask
