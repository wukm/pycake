#!/usr/bin/env python3

"""
This is the main program. It approximates the PCSVN of a list of samples.
It does not do network completion.

"""

from placenta import (get_named_placenta, cropped_args, cropped_view,
                      list_placentas, list_by_quality, open_typefile,
                      open_tracefile, add_ucip_to_mask, measure_ncs_markings)

from merging import nz_percentile, apply_threshold
from scoring import (compare_trace, rgb_to_widths, merge_widths_from_traces,
                     filter_widths, mcc, confusion, skeletonize_trace)

from pcsvn import extract_pcsvn, scale_label_figure, get_outname_lambda
from preprocessing import inpaint_hybrid

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import os.path
import os
import json
import datetime
import pandas

# for some post_processing, this needs to be moved elsewhere
from skimage.filters import sobel
from frangi import frangi_from_image
from plate_morphology import dilate_boundary
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import random_walker


# INITIALIZE SAMPLES ________________________________________________________
#   There are several ways to initialize samples. Uncomment one.

# load all 201 samples
#placentas = list_placentas('T-BN')
# load placentas from a certain quality category 0=good, 1=okay, 2=fair, 3=poor
placentas = list_by_quality(0, N=2)
placentas.extend(list_by_quality(1, N=1))
placentas.extend(list_by_quality(2, N=1))
placentas.extend(list_by_quality(3, N=1))

# load from a file (sample names are keys of the json file)
# placentas = list_by_quality(json_file='manual_batch.json')

# for a single named sample, use a 1 element list.
# placentas = ['T-BN0204423.png']

n_samples = len(placentas)

# RUNTIME OPTIONS ___________________________________________________________
#   Where to save and whether or not to use old targets.

MAKE_NPZ_FILES = False  # pickle frangi targets if you can
USE_NPZ_FILES = False   # use old npz files if you can
NPZ_DIR = 'output/181121-refactoring'  # where to look for npz files
OUTPUT_DIR = 'output/181121-hessian'  # where to save outputs

# add in a meta switch for verbosity (or levels)
#VERBOSE = False

# FRANGI / EXTRACT_PCSVN OPTIONS ____________________________________________

# Find bright curvilinear structure against a dark background -> True
# Find dark curvilinear structure against a bright background -> False
# DARK_BG -> ignore and return signed Frangi scores
DARK_BG = False

# Along with the above, this will return "opposite" signed frangi scores.
# if this is True, then DARK_BG controls the "polarity" of the filter.
# See frangi.get_frangi_targets for details.
SIGNED_FRANGI = False

# Do not calculate hessian scores close to the boundary (this is important
# mainly in terms of ensuring that the hessian is very large on the edge of
# the plate (which would influence gamma calculation)
DILATE_PER_SCALE = True

# Attempt to remove glare from sample (some are OK, some are bad)
REMOVE_GLARE = True

# What scales to use!
log_range = (-2, 3.5)
n_scales = 40

# when showing "large scales only", this is where to start
# (some index between 0 and n_scales)
LO_offset = 8

# Explicit Frangi Parameters (pass an array as long as scales or pass None)
betas = None  # None -> use default parameters (0.5)
gammas = None # None -> use default parameters (calculate half of hessian norm)
alphas = None # none to set later
fixed_alpha = .15


# Scoring Decisions (don't need to touch these)
ucip_radius = 90  # area around the umbilical cord insertion point to ignore

# some other initializations, don't mind me




# CODE BEGINS HERE ____________________________________________________________

n_samples = len(placentas)
scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
mccs = dict()  # empty dict to store MCC's of each sample
pncs = dict()  # empty dict to store percent network covered for each sample

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(n_samples, "samples total!")

for i, filename in enumerate(placentas):

    print('*'*80)
    print(f'extracting PCSVN of {filename}\t ({i} of {n_samples})')

    # --- Setup, Preprocessing, Frangi Filter (it's mixed up) -----------------

    raw_img = get_named_placenta(filename)

    if REMOVE_GLARE:
        img = inpaint_hybrid(raw_img)
    else:
        img = raw_img  # in case preprocessing happens in extract_pcsvn

    if USE_NPZ_FILES:
        # find the first npz file with the sample name in it in the
        # specified directory.
        stub = filename.rstrip('.png')
        for f in os.scandir(NPZ_DIR):
            if f.name.endswith('npz') and f.name.startswith(stub):
                npz_filename = os.path.join(NPZ_DIR, f.name)
                print(f'using the npz file {npz_filename}')
                break  # we'll just use the first one we can find.
        else:
            print(f'no npz file found for {filename}.')
            npz_filename = None
    else:
        npz_filename = None

    # set a lambda function to make output file names
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)

    if npz_filename is not None:

        F = np.load(npz_filename)['F']

        # in case preprocessing happens inside extract_pcsvn, do it out here

        print('successfully loaded the frangi targets!')

    else:
        print('finding multiscale frangi targets')

        # F is an array of frangi scores of shape (*img.shape, n_scales)
        F, jfile = extract_pcsvn(img, filename, dark_bg=DARK_BG, betas=betas,
                                 scales=scales, gammas=gammas,
                                 kernel='discrete', dilate_per_scale=True,
                                 verbose=False, signed_frangi=SIGNED_FRANGI,
                                 generate_json=True, output_dir=OUTPUT_DIR)

        if MAKE_NPZ_FILES:
            npzfile = ".".join((outname("F").rsplit('.', maxsplit=1)[0], 'npz')
                               )
            print("saving frangi targets to ", npzfile)
            np.savez_compressed(npzfile, F=F)

    # --- Merging & Postprocessing --------------------------------------------

    # This is the maximum frangi response over all scales at each location
    Fmax = F.max(axis=-1)

    print("...making outputs")

    if alphas is None:
        print("thresholding alphas with top 5% scores at each scale")
        alphas = np.array([nz_percentile(F[:, :, k], 95.0)
                           for k in range(n_scales)]
                          )
    scale_maxes = np.array([F[...,i].max() for i in range(F.shape[-1])])
    #print('percentile alphas:', alphas)
    #print('max at each scale:', scale_maxes)
    table = pandas.DataFrame(np.dstack((scales, alphas, scale_maxes)).squeeze(),
                                columns=('σ', 'α_p', 'max(F_σ)'))

    print(table)
    # threshold the responses at each of these values and get labels of max
    approx, labs = apply_threshold(F, alphas, return_labels=True)

    # --- Scoring and Outputs -------------------------------------------------

    # get the main (boolean) tracefile and the RGB tracefiles
    trace = open_tracefile(filename, as_binary=True)
    A_trace = open_typefile(filename, 'arteries')
    V_trace = open_typefile(filename, 'veins')
    skeltrace = skeletonize_trace(A_trace, V_trace)

    # get a matrix of pixel widths in the trace
    widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')

    # find cord insertion point and resolution of the image
    ucip_midpoint, resolution = measure_ncs_markings(filename=filename)

    # if verbose:
    # print(f"The umbilicial cord insertion point is at {ucip_midpoint}")
    # print(f"The resolution of the image is {resolution} pixels per cm.")

    # mask anywhere close to the UCIP
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=int(ucip_radius),
                                 mask=img.mask)

    # The following are examples of things you can do:

    # matrix of widths of traced image
    # min_widths = merge_widths_from_traces(A_trace, V_trace,
    #                                       strategy='minimum')

    # trace ignoring largest vessels (19 pixels wide)
    # trace_smaller_only = filter_widths(min_widths, min_width=3, max_width=17)
    # trace_smaller_only != 0

    # use only some scales
    #approx_LO, labs_LO = apply_threshold(F[:,:, LO_offset:], alphas[LO_offset:])
    approx_FA, labs_FA = apply_threshold(F, fixed_alpha)

    # fix labels to incorporate offset
    #labs_LO = (labs_LO != 0)*(labs_LO + LO_offset)

    # confusion matrix against default trace
    confuse = confusion(approx, trace, bg_mask=ucip_mask)
    #confuse_LO = confusion(approx_LO, trace, bg_mask=ucip_mask)
    confuse_FA = confusion(approx_FA, trace, bg_mask=ucip_mask)

    m_score, counts = mcc(approx, trace, ucip_mask, return_counts=True)
    m_score_FA, counts_FA = mcc(approx_FA, trace, ucip_mask,
                                return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts # return these for more analysis?

    total = np.invert(ucip_mask).sum()
    print(f'TP: {TP}\t TN: {TN}\nFP: {FP}\tFN: {FN}')
    # just a sanity check
    print(f'TP+TN+FP+FN={TP+TN+FP+FN}\ttotal pixels={total}')

    # MOVE THIS ELSEWHERE
    s = sobel(img)
    s = dilate_boundary(s, mask=img.mask, radius=20)
    finv = frangi_from_image(s, sigma=0.8, dark_bg=True, dilation_radius=10)
    finv_thresh = nz_percentile(finv, 80)
    margins = remove_small_objects((finv > finv_thresh).filled(0), min_size=32)
    margins_added = np.logical_or(margins, approx)
    margins_added = remove_small_holes(margins_added, area_threshold=100,
                                       connectivity=2)

    confuse_margins = confusion(margins_added, trace, bg_mask=ucip_mask)

    # random walker markers
    markers = np.zeros(img.shape, dtype=np.uint8)
    markers[Fmax < .1] = 1
    markers[margins_added] = 2
    rw = random_walker(img, markers, beta=1000)
    approx_rw = (rw == 2)
    confuse_rw = confusion(approx_rw, trace, bg_mask=ucip_mask)
    m_score_rw, counts_rw = mcc(approx_rw, trace, ucip_mask,
                                return_counts=True)
    pnc_rw = np.logical_and(skeltrace, approx_rw).sum() / skeltrace.sum()

    mccs[filename] =  (m_score, m_score_FA, m_score_rw)

    print(f'mcc score of {m_score:.3} for {filename}')
    #print(f'mcc score of {m_score_LO:.3} with larger sigmas only')
    print(f'mcc score of {m_score_rw:.3} after random walker')
    # --- Generating Visual Outputs--------------------------------------------
    crop = cropped_args(img)  # these indices crop out the mask significantly

    # save the raw, unaltered image
    plt.imsave(outname('0_raw'), raw_img[crop].filled(0), cmap=plt.cm.gray)

    # save the preprocessed image
    plt.imsave(outname('1_img'), img[crop].filled(0), cmap=plt.cm.gray)

    # save the maximum frangi output over all scales
    plt.imsave(outname('2_fmax'), Fmax[crop], vmin=0, vmax=1.0,
               cmap=plt.cm.nipy_spectral)

    # only save the colorbar the first time
    save_colorbar = (i==0)
    scale_label_figure(labs, scales, crop=crop,
                       savefilename=outname('3_labeled'), image_only=True,
                       save_colorbar_separate=save_colorbar,
                       output_dir=OUTPUT_DIR)

    plt.imsave(outname('4_confusion'), confuse[crop])

    #plt.imsave(outname('7_confusion_LO'), confuse_LO[crop])
    plt.imsave(outname('7_confusion_FA'), confuse_FA[crop])
    plt.imsave(outname('A_confusion_rw'), confuse_rw[crop])

    plt.imsave(outname('9_margin_for_rw'), confuse_margins[crop])
    percent_covered = np.logical_and(skeltrace, approx).sum() / skeltrace.sum()
    percent_covered_FA = np.logical_and(skeltrace,
                                        approx_FA).sum() / skeltrace.sum()

    pncs[filename] = (percent_covered, percent_covered_FA, pnc_rw)



    st_colors = {
        'TN': (79,79,79),  # true negative# 'f7f7f7'
        'TP': (0, 0, 0),  # true positive  # '000000'
        'FN': (201,53,108),  # false negative # 'f1a340' orange
        'FP': (92,92,92),  # false positive
        'mask': (247, 200, 200)  # mask color (not used in MCC calculation)
    }

    print('percentage of skeltrace covered:', f'{percent_covered:.2%}')
    print('percentage of skeltrace covered (larger sigmas only):',
          f'{percent_covered_FA:.2%}')
    print('percentage of skeltrace covered (random_walker):',
          f'{pnc_rw:.2%}')
    plt.imsave(outname('5_coverage'), confusion(approx, skeltrace,
                                                colordict=st_colors)[crop])
    #plt.imsave(outname('8_coverage_LO'), confusion(approx_LO, skeltrace)[crop])
    plt.imsave(outname('8_coverage_FA'), confusion(approx_FA, skeltrace,
                                                   colordict=st_colors)[crop])
    plt.imsave(outname('B_coverage_rw'), confusion(approx_rw, skeltrace,
                                                   colordict=st_colors)[crop])

    # make the graph that shows what scale the max was pulled from

    scale_label_figure(labs_FA, scales, crop=crop,
                       savefilename=outname('6_labeled_FA'), image_only=True,
                       save_colorbar_separate=False, output_dir=OUTPUT_DIR)
    plt.close('all')  # something's leaking :(


    ### THIS IS ALL A HORRIBLE MESS. FIX IT

    # why don't you just return the dict instead
    with open(jfile, 'r') as f:
        slog = json.load(f)

    c2d = lambda t: dict(zip(('TP','TN', 'FP', 'FN'), [int(c) for c in t]))

    slog['counts'] = c2d(counts)
    slog['counts_FA'] = c2d(counts_FA)
    slog['counts_rw'] = c2d(counts_rw)
    slog['pnc'] = pncs[filename]
    slog['mcc'] = mccs[filename]
    slog['scale_maxes'] = list(scale_maxes)
    slog['alphas'] = list(alphas)

    with open(jfile, 'w') as f:
        json.dump(slog, f)

# Post-run Meta-Output and Logging ____________________________________________

timestring = datetime.datetime.now()
timestring = timestring.strftime("%y%m%d_%H%M")

mccfile = os.path.join(OUTPUT_DIR, f"runlog_{timestring}.json")

runlog = {
    'time': timestring,
    'dark_bg': DARK_BG,
    'dilate_per_scale': DILATE_PER_SCALE,
    'log_range': log_range,
    'n_scales': n_scales,
    'scales': list(scales),
    'alphas': list(alphas),
    'betas': None,
    'use_npz_files': False,
    'remove_glare': REMOVE_GLARE,
    'files': list(placentas),
    'MCCS': mccs,
    'PNC': pncs
}

# save to a json file
with open(mccfile, 'w') as f:
    json.dump(runlog, f, indent=True)
