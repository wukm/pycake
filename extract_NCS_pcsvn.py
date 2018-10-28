#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from get_placenta import get_named_placenta, cropped_args, cropped_view
from get_placenta import list_placentas, open_typefile, list_by_quality

from score import compare_trace

from pcsvn import extract_pcsvn, scale_label_figure, apply_threshold
from pcsvn import get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

from hfft import fft_gradient
from score import mcc

import os.path
import os
import json
import datetime

#placentas = list_placentas('T-BN') # load allllll placentas
placentas = list_by_quality(2)
#placentas.extend(list_by_quality(1))

# obviously need to give this function a more descriptive name
#placentas = list_by_quality(json_file='manual_batch.json')

n_samples = len(placentas)

OUTPUT_DIR = 'output/181023-morescales'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DARK_BG = False
log_range = (-4, 4.5)
n_scales = 30
scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
#alphas = scales**(2/3) / scales[-1]
alphas = [0.15 for s in scales]
betas = None
print(n_samples, "samples total!")

m_scores = dict()

for i, filename in enumerate(placentas):
    print('*'*80)
    print('extracting PCSVN of', filename,
            '\t ({} of {})'.format(i,n_samples))
    F, img, _ , _ = extract_pcsvn(filename, DARK_BG=DARK_BG,
                                alpha=.1, alphas=alphas, betas=betas,
                                scales=scales,  log_range=log_range,
                                verbose=False, generate_graphs=False,
                                   n_scales=n_scales, generate_json=True,
                                           output_dir=OUTPUT_DIR,
                                  kernel='discrete')

    crop = cropped_args(img)
    print("...making outputs")
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)


    approx, labs = apply_threshold(F, alphas, return_labels=True)

    confusion = compare_trace(approx, filename=filename)
    trace = open_typefile(filename, 'trace').astype('bool')
    trace = np.invert(trace)

    m_score, counts = mcc(approx, trace, img.mask, return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts

    total = np.invert(img.mask).sum()
    print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    print('TP+TN+FP+FN={}\ntotal pixels={}'.format(TP+TN+FP+FN,total))

    print("MCC for {}:\t".format(filename), m_score)

    m_scores[filename] =  m_score

    plt.imsave(outname('0_raw'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('1_confusion'), confusion[crop])
    scale_label_figure(labs, scales, crop=crop, savefilename=outname('2_labeled'),
                       image_only=False)
    plt.imsave(outname('3_fmax'), F.max(axis=-1)[crop],
               vmin=0,vmax=0.5,
               cmap=plt.cm.nipy_spectral)
    plt.close('all') # something's leaking :(
# json file with mccs
timestring = datetime.datetime.now()
timestring = timestring.strftime("%y%m%d_%H%M")
mccfile = os.path.join(OUTPUT_DIR, f"MCCS_{timestring}.json")

with open(mccfile, 'w') as f:
    json.dump(m_scores, f, indent=True)
