#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from get_placenta import get_named_placenta, cropped_args, cropped_view
from get_placenta import list_placentas

from score import compare_trace

from pcsvn import extract_pcsvn, scale_label_figure, apply_threshold
from pcsvn import get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

#filename = 'T-BN0033885.png'
placentas = list_placentas('T-BN')

DARK_BG = False
log_range = (-2, 4)

for i, filename in enumerate(placentas):

    F, img, scales = extract_pcsvn(filename, DARK_BG=DARK_BG,
                                alpha=None, log_range=log_range,
                                verbose=False, generate_graphs=False)

    crop = cropped_args(img)

    outname = get_outname_lambda(filename, output_dir='output/newalpha')

    alphas = scales**(2/3) / scales[-1]

    trace, labs = apply_threshold(F, alphas, return_labels=True)
    scale_label_figure(labs, scales, crop=crop, savefilename=outname('labeled'))

    confusion = compare_trace(trace, filename=filename)
    plt.imsave(outname('confusion'), confusion[crop])

    if i > 10:
        break
