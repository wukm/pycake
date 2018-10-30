#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import imread
CYAN = [0,255,255]
YELLOW = [255,255,0]
# given the image img (make sure no alpha channel)
# find all cyan pixels (there are two boxes of 3 pixels each and we
# just want to extract the middle of each
print('the image size is', img.shape)
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
radius = X.max() - X.min() // 2

M = (X.min() + radius, Y.min() + radius)
print('the middle of the UCIP location is', M)
print('the radius outward is', radius)
print('the total measurable diameter is', radius*2 + 1)


