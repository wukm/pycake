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
n_samples = len(placentas)

OUTPUT_DIR = 'output/newalpha'
DARK_BG = False
log_range = (-2, 4.5)
n_scales = 20
scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
alphas = scales**(2/3) / scales[-1]

print(N_samples, "samples total!")
for i, filename in enumerate(placentas):
    print('*'*80)
    print('extracting PCSVN of', filename,
            '\t ({} of {})'.format(i,N_samples))
    F, img, _ , _ = extract_pcsvn(filename, DARK_BG=DARK_BG,
                                alpha=.1, alphas=alphas,
                                scales=scales,  log_range=log_range,
                                verbose=False, generate_graphs=False,
                                   n_scales=n_scales, generate_json=True,
                                           output_dir=OUTPUT_DIR)


    crop = cropped_args(img)

    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)


    trace, labs = apply_threshold(F, alphas, return_labels=True)
    scale_label_figure(labs, scales, crop=crop, savefilename=outname('2_labeled'),
                       image_only=True)

    confusion = compare_trace(trace, filename=filename)
    plt.imsave(outname('1_confusion'), confusion[crop])

    plt.imsave(outname('0_raw'), img[crop].filled(0), cmap=plt.cm.gray)
