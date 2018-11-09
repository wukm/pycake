#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from get_placenta import get_named_placenta, cropped_args, cropped_view
from get_placenta import list_placentas, open_typefile, list_by_quality, open_tracefile

from score import compare_trace

from pcsvn import extract_pcsvn, scale_label_figure, apply_threshold
from pcsvn import get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

from score import mcc

import os.path
import os
import json
import datetime

placentas = list_by_quality(0)
#placentas = list_placentas('T-BN') # load allllll placentas
#placentas = list_by_quality(json_file='manual_batch.json')

n_samples = len(placentas)

OUTPUT_DIR = 'output/181104-refactoring'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DARK_BG = False
DILATE_PER_SCALE = True
SIGNED_FRANGI = True
log_range = (-4, 5.5)
n_scales = 40
scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
alphas = [0.15 for s in scales]
betas = None # will be given default parameters
gammas =  None # will be given default parameters
print(n_samples, "samples total!")

mccs = dict() # where to score MCC's of each sample

for i, filename in enumerate(placentas):
    print('*'*80)
    print('extracting PCSVN of', filename,
            '\t ({} of {})'.format(i,n_samples))
    F, img, _ , _ = extract_pcsvn(filename, DARK_BG=DARK_BG,
                                alphas=alphas, betas=betas, scales=scales,
                                  gammas=gammas,
                                kernel='discrete', dilate_per_scale=True,
                                verbose=False, generate_graphs=False,
                                signed_frangi=SIGNED_FRANGI,
                                generate_json=True, output_dir=OUTPUT_DIR)

    crop = cropped_args(img) # these make viewing easier
    print("...making outputs")
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)

    approx, labs = apply_threshold(F, alphas, return_labels=True)


    trace = open_tracefile(filename, as_binary=True)

    # open up RGB tracefiles and convert to widths
    A_trace = open_typefile(filename, 'arteries')
    V_trace = open_typefile(filename, 'veins')

    # confusion matrix against default
    confuse = compare_trace(approx, trace=trace)

    m_score, counts = mcc(approx, trace, img.mask, return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts # return these for more analysis?

    #total = np.invert(img.mask).sum()
    #print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    #print('TP+TN+FP+FN={}\ntotal pixels={}'.format(TP+TN+FP+FN,total))

    mccs[filename] =  m_score

    plt.imsave(outname('0_raw'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('1_confusion'), confusion[crop])

    # make the graph that shows what scale the max was pulled from
    scale_label_figure(labs, scales, crop=crop,
                       savefilename=outname('2_labeled'), image_only=False)

    # save the maximum frangi output
    plt.imsave(outname('3_fmax'), F.max(axis=-1)[crop],
               vmin=0,vmax=0.5,
               cmap=plt.cm.nipy_spectral)
    plt.close('all') # something's leaking :(
    break

# json file with mccs and other runtime info
timestring = datetime.datetime.now()
timestring = timestring.strftime("%y%m%d_%H%M")


mccfile = os.path.join(OUTPUT_DIR, f"runlog_{timestring}.json")

runlog = {
    'time': timestring,
    'DARK_BG': DARK_BG,
    'dilate_per_scale': DILATE_PER_SCALE,
    'log_range': log_range,
    'n_scales': n_scales,
    'scales': list(scales),
    'alphas': list(alphas),
    'betas': None,
    'files': list(placentas),
    'MCCS': mccs
}

with open(mccfile, 'w') as f:
    json.dump(runlog, f, indent=True)
