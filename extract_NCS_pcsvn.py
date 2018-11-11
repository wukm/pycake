#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from placenta import get_named_placenta, cropped_args, cropped_view
from placenta import list_placentas, list_by_quality
from placenta import open_typefile, open_tracefile
from placenta import add_ucip_to_mask, measure_ncs_markings

from score import compare_trace, rgb_to_widths, merge_widths_from_traces
from score import filter_widths, mcc, confusion

from pcsvn import extract_pcsvn, scale_label_figure, apply_threshold
from pcsvn import get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import os.path
import os
import json
import datetime

#different ways to initialize samples
#placentas = list_by_quality(0)
placentas = list_placentas('T-BN')  # load allllll placentas
#placentas = list_by_quality(json_file='manual_batch.json')

n_samples = len(placentas)

MAKE_NPZ_FILES = False # pickle frangi targets if you can
USE_NPZ_FILES = True  # use old npz files if you can
NPZ_DIR = 'output/181108-bigrun'

OUTPUT_DIR = 'output/181110-onepercent'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DARK_BG = False
DILATE_PER_SCALE = True
SIGNED_FRANGI = False
log_range = (-4, 5.5)
n_scales = 40

scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
alphas = [0.15 for s in scales]
betas = None  # will be given default parameters
gammas =  None  # will be given default parameters

mccs = dict()  # empty dict to store MCC's of each sample

print(n_samples, "samples total!")

for i, filename in enumerate(placentas):

    print('*'*80)
    print(f'extracting PCSVN of {filename}\t ({i} of {n_samples})')

    if USE_NPZ_FILES:
        # this needs major work. write better code.
        stub = filename.rstrip('.png')
        for f in os.scandir(NPZ_DIR):
            if f.name.endswith('npz') and f.name.startswith(stub):
                npz_filename = os.path.join(NPZ_DIR, f.name)
                print(f'using the npz file {npz_filename}')
                break
        else:
            print(f'no npz file found for {filename}.')
            npz_filename = None

    # set a lambda function to make output file names
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)

    if npz_filename is not None:
        F = np.load(npz_filename)['F']
        img = get_named_placenta(filename, maskfile=None)
        print('successfully loaded the frangi targets!')

    else:
        print('finding multiscale frangi targets')

        F, img = extract_pcsvn(filename, DARK_BG=DARK_BG, alphas=alphas,
                            betas=betas, scales=scales, gammas=gammas,
                            kernel='discrete', dilate_per_scale=True,
                            verbose=False, signed_frangi=SIGNED_FRANGI,
                            generate_json=True, output_dir=OUTPUT_DIR)

        if MAKE_NPZ_FILES:
            npzfile = ".".join((outname("F").rsplit('.', maxsplit=1)[0],'npz'))
            print("saving frangi targets to ", npzfile)
            np.savez_compressed(npzfile, F=F)

    crop = cropped_args(img) # these indices crop out the mask significantly

    print("...making outputs")

    # get the 99% frangi filter score
    print("rewriting alphas with 1% scores")
    p_alphas = [np.percentile(F[:,:,k],99.0) for k in range(n_scales)]
    alphas = np.array(p_alphas)

    approx, labs = apply_threshold(F, alphas, return_labels=True)

    # get the main tracefile
    trace = open_tracefile(filename, as_binary=True)

    ucip_midpoint, resolution = measure_ncs_markings(filename=filename)

    print(f"The umbilicial cord insertion point is at {ucip_midpoint}")
    print(f"The resolution of the image is {resolution} pixels per cm.")

    # mask anywhere close to (within 90px L2 distance of) the UCIP
    # this is imperically how the it is, although could be bigger
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=90, mask=img.mask)

    # open up RGB tracefiles (keep for visualizing?)
    A_trace = open_typefile(filename, 'arteries')
    V_trace = open_typefile(filename, 'veins')

    # matrix of widths of traced image
    widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')
    min_widths = merge_widths_from_traces(A_trace, V_trace, strategy='minimum')

    # trace ignoring largest vessels (19 pixels wide)
    trace_smaller_only = filter_widths(min_widths, min_width=3, max_width=17)
    trace_smaller_only != 0
    # use limited scales
    approx_larger_only = F[:,:, 24:].max(axis=-1) > .15
    # confusion matrix against default trace
    confuse = confusion(approx, trace, bg_mask=ucip_mask)

    m_score, counts = mcc(approx, trace, ucip_mask, return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts # return these for more analysis?

    total = np.invert(ucip_mask).sum()
    print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    print('TP+TN+FP+FN={}\ttotal pixels={}'.format(TP+TN+FP+FN,total))

    mccs[filename] =  m_score

    print(f'mcc score of {m_score} for {filename}')

    plt.imsave(outname('0_raw'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('1_confusion'), confuse[crop])

    # make the graph that shows what scale the max was pulled from
    scale_label_figure(labs, scales, crop=crop,
                       savefilename=outname('2_labeled'), image_only=True,
                       save_colormap_separate=True, output_dir=OUTPUT_DIR)

    # save the maximum frangi output
    plt.imsave(outname('3_fmax'), F.max(axis=-1)[crop],
               vmin=0,vmax=1.0,
               cmap=plt.cm.nipy_spectral)
    plt.close('all') # something's leaking :(


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
