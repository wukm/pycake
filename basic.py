#!/usr/bin/env python3

"""
THIS IS A BASIC SET UP TO EXPLORE, PLAY AROUND A BIT
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)
