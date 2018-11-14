#!/usr/bin/env python3

"""
This is the main program. It approximates the PCSVN


"""

from placenta import (get_named_placenta, cropped_args, cropped_view,
                      list_placentas, list_by_quality, open_typefile,
                      open_tracefile, add_ucip_to_mask, measure_ncs_markings)

from merging import nz_percentile, apply_threshold
from score import (compare_trace, rgb_to_widths, merge_widths_from_traces,
                   filter_widths, mcc, confusion)

from pcsvn import extract_pcsvn, scale_label_figure, get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import os.path
import os
import json
import datetime

# INITIALIZE SAMPLES ________________________________________________________

# initialize a list of samples (several different ways)
#placentas = list_by_quality(0)
placentas = list_placentas('T-BN')  # load allllll placentas
#placentas = list_by_quality(json_file='manual_batch.json')
#placentas = ['T-BN1662406.png'] # for a single sample, use a 1 element list.

n_samples = len(placentas)

# RUNTIME OPTIONS ___________________________________________________________

MAKE_NPZ_FILES = True # pickle frangi targets if you can
USE_NPZ_FILES = True  # use old npz files if you can
NPZ_DIR = 'output/181112-bigrun' # where to look for npz files
OUTPUT_DIR = 'output/181112-bigrun' # where to save outputs


# EXTRACT_PCSVN OPTIONS _____________________________________________________

# find bright curvilinear structure against a dark background -> True
# find dark curvilinear structure against a bright background -> False
# DARK_BG -> ignore and return signed Frangi scores
DARK_BG = False

# along with the above, this will return "opposite" signed frangi scores.
# if this is True, then DARK_BG controls the "polarity" of the filter.
# See frangi.get_frangi_targets for details.
SIGNED_FRANGI = False

# do not calculate hessian scores close to the boundary (this is important
# mainly in terms of ensuring that the hessian is very large on the edge of
# the plate (which would influence gamma calculation)
DILATE_PER_SCALE = True

log_range = (-3, 5.5)
n_scales = 40


scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
alphas = [0.15 for s in scales]
betas = None  # will be given default parameters
gammas =  None  # will be given default parameters

mccs = dict()  # empty dict to store MCC's of each sample

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(n_samples, "samples total!")

for i, filename in enumerate(placentas):

    print('*'*80)
    print(f'extracting PCSVN of {filename}\t ({i} of {n_samples})')

    if USE_NPZ_FILES:
        # find the first npz file with the sample name in it in the
        # specified directory.
        stub = filename.rstrip('.png')
        for f in os.scandir(NPZ_DIR):
            if f.name.endswith('npz') and f.name.startswith(stub):
                npz_filename = os.path.join(NPZ_DIR, f.name)
                print(f'using the npz file {npz_filename}')
                break # just use the first one.
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

    p_alphas = [nz_percentile(F[:,:,k],95.0) for k in range(n_scales)]
    alphas = np.array(p_alphas)

    approx, labs = apply_threshold(F, alphas, return_labels=True)

    # get the main tracefile
    trace = open_tracefile(filename, as_binary=True)

    ucip_midpoint, resolution = measure_ncs_markings(filename=filename)

    #print(f"The umbilicial cord insertion point is at {ucip_midpoint}")
    #print(f"The resolution of the image is {resolution} pixels per cm.")

    # mask anywhere close to (within 90px L2 distance of) the UCIP
    # this is imperically how the it is, although could be bigger
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=90, mask=img.mask)

    # open up RGB tracefiles (keep for visualizing?)
    A_trace = open_typefile(filename, 'arteries')
    V_trace = open_typefile(filename, 'veins')

    # matrix of widths of traced image
    widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')
    #min_widths = merge_widths_from_traces(A_trace, V_trace, strategy='minimum')

    # trace ignoring largest vessels (19 pixels wide)
    #trace_smaller_only = filter_widths(min_widths, min_width=3, max_width=17)
    #trace_smaller_only != 0
    # use limited scales
    approx_LO, labs_LO = apply_threshold(F[:,:, 24:],
                                                           alphas[24:])
    # confusion matrix against default trace
    confuse = confusion(approx, trace, bg_mask=ucip_mask)
    confuse_LO = confusion(approx_LO, trace,
                                    bg_mask=ucip_mask)

    m_score, counts = mcc(approx, trace, ucip_mask, return_counts=True)
    m_score_LO, counts_LO = mcc(approx_LO, trace,
                                                  ucip_mask, return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts # return these for more analysis?

    total = np.invert(ucip_mask).sum()
    print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    print('TP+TN+FP+FN={}\ttotal pixels={}'.format(TP+TN+FP+FN,total))

    mccs[filename] =  (m_score, m_score_LO)

    print(f'mcc score of {m_score} for {filename}')
    print(f'mcc score of {m_score_LO} with larger sigmas only')

    plt.imsave(outname('6_raw'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('3_confusion'), confuse[crop])
    plt.imsave(outname('4_confusion_LO'), confuse_LO[crop])

    # make the graph that shows what scale the max was pulled from
    scale_label_figure(labs, scales, crop=crop,
                       savefilename=outname('2_labeled'), image_only=True,
                       save_colormap_separate=True, output_dir=OUTPUT_DIR)

    scale_label_figure(labs_LO, scales, crop=crop,
                       savefilename=outname('5_labeled'), image_only=True,
                       save_colormap_separate=False, output_dir=OUTPUT_DIR)
    # save the maximum frangi output
    plt.imsave(outname('1_fmax'), F.max(axis=-1)[crop],
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
