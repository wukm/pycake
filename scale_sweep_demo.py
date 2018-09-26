#!/usr/bin/env python3

from get_placenta import get_named_placenta
from frangi import frangi_from_image

import numpy as np
import numpy.ma as ma

import os.path
import matplotlib.pyplot as plt

frangi_from_image(img, sigma, beta=0.5, gamma=None, dark_bg=False,
                  dilation_radius=None, threshold=None, return_debug_info=False)
