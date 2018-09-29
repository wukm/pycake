#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from get_placenta import get_named_placenta, cropped_args, cropped_view
from get_placenta import list_placentas

from score import compare_trace

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import os.path

#placenta = list_placentas('T-BN')[0]
filename = get_named_placenta('T-BN0033885.png')




