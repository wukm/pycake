#!/usr/bin/env python3

"""
Most of this is obsolete. This was the main progam for approximating
PCSVN network. It is no longer used mostly because there are too many methods
of segmentation with no clear victor. the script that replaces this one is
probably compare_segmenations

"""

from placenta import (get_named_placenta, cropped_args, cropped_view,
                      list_placentas, list_by_quality, open_typefile,
                      open_tracefile, add_ucip_to_mask, measure_ncs_markings)

from merging import nz_percentile, apply_threshold, sieve_scales, view_slices

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
from plate_morphology import dilate_boundary, mask_cuts_simple
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import random_walker
from postprocessing import random_walk_fill, random_walk_scalewise


# INITIALIZE SAMPLES ________________________________________________________
#   There are several ways to initialize samples. Uncomment one.

# load all 201 samples
# placentas = list_placentas('T-BN')
# load placentas from a certain quality category 0=good, 1=okay, 2=fair, 3=poor

#placentas = list_by_quality(2)
#placentas.extend(list_by_quality(3))

placentas = list_by_quality(0, N=1)
# load from a file (sample names are keys of the json file)
# placentas = list_by_quality(json_file='manual_batch.json')

# for a single named sample, use a 1 element list.
# placentas = ['T-BN0204423.png']

#placentas = ['barium1.png',]
# RUNTIME OPTIONS ___________________________________________________________
#   Where to save and whether or not to use old targets.

MAKE_NPZ_FILES = False # pickle frangi targets if you can
USE_NPZ_FILES = False # use old npz files if you can
NPZ_DIR = 'output/181204-test'  # where to look for npz files
OUTPUT_DIR = 'output/181204-test'  # where to save outputs

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
SIGNED_FRANGI = True

# Do not calculate hessian scores close to the boundary (this is important
# mainly in terms of ensuring that the hessian is very large on the edge of
# the plate (which would influence gamma calculation)
DILATE_PER_SCALE = True

# Attempt to remove glare from sample (some are OK, some are bad)
FLATTEN_MODE = 'L' # 'G' or 'L'
REMOVE_GLARE = True

# Which scales to use
SCALE_RANGE = (-1.5, 3.2); SCALE_TYPE = 'logarithmic'
#SCALE_RANGE = (.2, 12); SCALE_TYPE = 'linear'
N_SCALES = 20

# use this if you want to use a custom argument (comment out the above)
SCALES = None
#SCALE_RANGE = None, SCALE_TYPE == 'custom'


# Explicit Frangi Parameters (pass a scalar, array as long as scales)
BETAS = 0.35
GAMMAS = 0.5
CS = None # pass scalar, array, or None
ALPHAS = None # set custom alphas or calculate later
FIXED_ALPHA = .4

RESCALE_FRANGI = True
GRADIENT_FILTER = False


# Scoring Decisions (don't need to touch these)
UCIP_RADIUS = 60  # area around the umbilical cord insertion point to ignore




# CODE BEGINS HERE ____________________________________________________________

if SCALES is None:
    if SCALE_TYPE == 'linear':
        scales = np.linspace(*SCALE_RANGE, num=N_SCALES)
    elif SCALE_TYPE == 'logarithmic':
        scales = np.logspace(*SCALE_RANGE, num=N_SCALES, base=2)
else:
    scales = SCALES
    SCALE_TYPE = 'custom'  # this and the next three lines are just for logging
    N_SCALES = len(SCALES)
    SCALES = (min(SCALES), max(SCALES))

mccs = dict()  # empty dict to store MCC's of each sample
pncs = dict()  # empty dict to store percent network covered for each sample
precisions = dict()

n_samples = len(placentas)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(n_samples, "samples total!")

for i, filename in enumerate(placentas):

    print('*'*80)
    print(f'extracting PCSVN of {filename}\t ({i} of {n_samples})')

    # --- Setup, Preprocessing, Frangi Filter (it's mixed up) -----------------

    raw_img = get_named_placenta(filename, mode=FLATTEN_MODE)
    cimg = open_typefile(filename, 'raw')

    ucip = open_typefile(filename, 'ucip')

    if REMOVE_CUTS:
        #img, has_cut = mask_cuts_simple(raw_img, ucip, return_success=True)
        #img.data[img.mask] = 0 # actually zero out that area
        print("removing cuts doesn't do anything anymore")
        pass
    else:
        img = raw_img.copy()

    if REMOVE_GLARE:
        img = inpaint_hybrid(img)


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

        # F is an array of frangi scores of shape (*img.shape, N_SCALES)
        F, jfile = extract_pcsvn(img, filename, dark_bg=DARK_BG, beta=BETAS,
                                 scales=scales, gamma=GAMMAS, c=CS,
                                 kernel='discrete', dilate_per_scale=True,
                                 verbose=False, signed_frangi=SIGNED_FRANGI,
                                 generate_json=True, output_dir=OUTPUT_DIR,
                                 rescale_frangi=RESCALE_FRANGI,
                                 gradient_filter=GRADIENT_FILTER)

        if MAKE_NPZ_FILES:
            npzfile = ".".join((outname("F").rsplit('.', maxsplit=1)[0], 'npz')
                               )
            print("saving frangi targets to ", npzfile)
            np.savez_compressed(npzfile, F=F)

    # - Multiscale Analysis, etc --------------------------------------------

    # This is the maximum frangi response over all scales at each location
    Fmax = F.max(axis=-1)

    print("...making outputs")

    if ALPHAS is None:
        print("thresholding ALPHAS with top 5% scores at each scale")
        ALPHAS = np.array([nz_percentile(F[..., k], 95.0)
                           for k in range(N_SCALES)]
                          )
    # the maximum value of the entire image at each scale
    scale_maxes = np.array([F[...,i].max() for i in range(F.shape[-1])])

    table = pandas.DataFrame(np.dstack((scales, ALPHAS, scale_maxes)).squeeze(),
                                columns=('σ', 'α_p', 'max(F_σ)'))

    print(table)
    # threshold the responses at each of these values and get labels of max

    # --- Segmentation Postprocessing -------------------------------------------------

    # get the main (boolean) tracefile and the RGB tracefiles
    trace = open_tracefile(filename, as_binary=True)
    A_trace = open_typefile(filename, 'arteries')
    if A_trace is None:
        # there are no special trace files for this sample
        skeltrace = skeletonize_trace(trace)
    else:
        V_trace = open_typefile(filename, 'veins')
        skeltrace = skeletonize_trace(A_trace, V_trace)

        # get a matrix of pixel widths in the trace
        widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')

    # find cord insertion point and resolution of the image
    ucip_midpoint, resolution = measure_ncs_markings(ucip)
    # if verbose:
    # print(f"The umbilicial cord insertion point is at {ucip_midpoint}")
    # print(f"The resolution of the image is {resolution} pixels per cm.")

    if ucip_midpoint is None:
        ucip_mask = img.mask
    # mask anywhere close to the UCIP
    else:
        ucip_mask = add_ucip_to_mask(ucip_midpoint,
                                     radius=int(UCIP_RADIUS), mask=img.mask)

    # The following are examples of things you can do:

    # matrix of widths of traced image
    # min_widths = merge_widths_from_traces(A_trace, V_trace,
    #                                       strategy='minimum')

    # trace ignoring largest vessels (19 pixels wide)
    # trace_smaller_only = filter_widths(min_widths, min_width=3, max_width=17)
    # trace_smaller_only != 0


    # -- Segmentation Strategies ---------------------------------------------

    # strawman
    from skimage.filters import threshold_mean
    from functools import partial

    confusion_matrix = partial(confusion, truth=trace, bg_mask=ucip_mask)
    mcc_with_counts = partial(mcc, truth=trace, bg_mask=ucip_mask,
                              return_counts=True)
    percent_network_coverage = lambda a: np.sum(skeltrace&a)/np.sum(skeltrace)
    precision = lambda t: int(t[0]) / int(t[0] + t[2])
    V = np.transpose(F, axes=(2, 0, 1))


    approx_sm = threshold_mean(img.filled(img.compressed().mean()))
    mcc_sm, counts_sm = mcc_with_counts(approx_sm)
    prec_sm = precision(counts_sm)
    confuse_sm = confusion_matrix(approx_sm)
    pnc_sm = percent_network_coverage(approx_sm)

    approx_PF, labs_PF = apply_threshold(F, ALPHAS, return_labels=True)
    mcc_PF, counts_PF = mcc_with_counts(approx_PF)
    confuse_PF = confusion_matrix(approx_PF)
    pnc_PF = percent_network_coverage(approx_PF)
    prec_PF = precision(counts_PF)

    approx_FA, labs_FA = apply_threshold(F, FIXED_ALPHA)
    mcc_FA, counts_FA = mcc(approx_FA, trace, ucip_mask, return_counts=True)
    confuse_FA = confusion_matrix(approx_FA)
    pnc_FA = percent_network_coverage(approx_FA)
    prec_FA = precision(counts_FA)

    approx_RW, labs_RW = random_walk_scalewise(F, FIXED_ALPHA, return_labels=True)
    confuse_RW = confusion_matrix(approx_RW)
    mcc_RW, counts_RW = mcc_with_counts(approx_RW)
    pnc_RW = percent_network_coverage(approx_RW)
    prec_RW = precision(counts_RW)

    sieved = sieve_scales(V, 98, 95)
    approx_S, labs_S = (sieved != 0), sieved
    confuse_S = confusion_matrix(approx_S)
    mcc_S, counts_S = mcc_with_counts(approx_S)
    pnc_S = percent_network_coverage(approx_S)
    prec_S = precision(counts_S)

    # PUT MARGIN ADD IN HERE
    #
    #
    #
    #

    mccs[filename] =  (mcc_PF, mcc_FA, mcc_RW, mcc_S, mcc_sm)
    pncs[filename] = (pnc_PF, pnc_FA, pnc_RW, pnc_S, pnc_sm)
    precisions[filename] = (prec_PF, prec_FA, prec_RW, prec_S, prec_sm)

    scoretable = pandas.DataFrame(np.vstack((mccs[filename], pncs[filename],
                                            precisions[filename])),
                                  columns=('PF', 'FA', 'RW', 'PS', 'SM'),
                                  index=('MCC', 'skel coverage', 'precision'))

    print(scoretable)
    print('\n\n')
    print(scoretable.to_latex())


    # use only some scales
    #approx_LO, labs_LO = apply_threshold(F[:,:, LO_offset:], ALPHAS[LO_offset:])
    # fix labels to incorporate offset
    #labs_LO = (labs_LO != 0)*(labs_LO + LO_offset)
    #confuse_LO = confusion(approx_LO, trace, bg_mask=ucip_mask)

    #TP, TN, FP, FN = counts

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    #total = np.sum(~ucip_mask)
    #print(f'TP: {TP}\t TN: {TN}\nFP: {FP}\tFN: {FN}')
    # just a sanity check
    #print(f'TP+TN+FP+FN={TP+TN+FP+FN}\ttotal pixels={total}')


    #approx_rw, markers, margins_added = random_walk_fill(img, Fmax, .3, .01,
    #                                                     DARK_BG)




    #view_slices(F[crop], axis=-1, scales=scales)

    # --- Generating Visual Outputs--------------------------------------------

    # cmap and the "set bad" argument / mask color
    SCALE_CMAP = ('plasma', (1,1,1,1))

    crop = cropped_args(img)  # these indices crop out the mask significantly

    fmax_colors = plt.cm.plasma
    fmax_colors.set_bad('k', 1)

    # save the raw, unaltered image
    plt.imsave(outname('0_raw'), cimg[crop])

    # save the preprocessed image
    plt.imsave(outname('1_img'), img[crop].filled(0), cmap=plt.cm.gray)

    # save the maximum frangi output over all scales
    plt.imsave(outname('2_fmax'), ma.masked_where(Fmax==0,Fmax)[crop], vmin=0,
               vmax=1.0, cmap=fmax_colors)

    # only save the colorbar the first time
    #save_colorbar = (i==0)
    #scale_label_figure(labs, scales, crop=crop,
    #                   savefilename=outname('3_labeled'), image_only=True,
    #                   save_colorbar_separate=save_colorbar,
    #                   basecolor=SCALE_CMAP[1], base_cmap=SCALE_CMAP[0],
    #                   output_dir=OUTPUT_DIR)

    #plt.imsave(outname('4_confusion'), confuse[crop])

    #scale_label_figure(labs_rw, scales, crop=crop,
    #                   savefilename=outname('A_labeled_rw'), image_only=True,
    #                   save_colorbar_separate=save_colorbar,
    #                   basecolor=SCALE_CMAP[1], base_cmap=SCALE_CMAP[0],
    #                   output_dir=OUTPUT_DIR)


    #plt.imsave(outname('7_confusion_FA'), confuse_FA[crop])
    #plt.imsave(outname('B_confusion_rw'), confuse_rw[crop])
    ##plt.imsave(outname('A_markers_rw'), markers[crop])
    ##plt.imsave(outname('9_margin_for_rw'), confuse_margins[crop])



    st_colors = {
        'TN': (79,79,79),  # true negative# 'f7f7f7'
        'TP': (0, 0, 0),  # true positive  # '000000'
        'FN': (201,53,108),  # false negative # 'f1a340' orange
        'FP': (92,92,92),  # false positive
        'mask': (247, 200, 200)  # mask color (not used in MCC calculation)
    }

    #plt.imsave(outname('5_coverage'), confusion(approx, skeltrace,
    #                                            colordict=st_colors)[crop])
    #plt.imsave(outname('8_coverage_FA'), confusion(approx_FA, skeltrace,
    #                                               colordict=st_colors)[crop])
    #plt.imsave(outname('C_coverage_rw'), confusion(approx_rw, skeltrace,
    #                                               colordict=st_colors)[crop])

    # make the graph that shows what scale the max was pulled from

    #scale_label_figure(labs_FA, scales, crop=crop,
    #                   savefilename=outname('6_labeled_FA'), image_only=True,
    #                   basecolor=SCALE_CMAP[1], base_cmap=SCALE_CMAP[0],
    #                   save_colorbar_separate=False, output_dir=OUTPUT_DIR)


    #scale_label_figure(labs_S, scales, crop=crop,
    #                   savefilename=outname('D_labeled_S'), image_only=True,
    #                   basecolor=SCALE_CMAP[1], base_cmap=SCALE_CMAP[0],
    #                   save_colorbar_separate=False, output_dir=OUTPUT_DIR)

    #plt.imsave(outname('E_confusion_S'), confuse_S[crop])




    ### THIS IS ALL A HORRIBLE MESS. FIX IT
    # why don't you just return the dict instead
    with open(jfile, 'r') as f:
        slog = json.load(f)

    c2d = lambda t: dict(zip(('TP','TN', 'FP', 'FN'), [int(c) for c in t]))

    slog['counts'] = c2d(counts)
    slog['counts_FA'] = c2d(counts_FA)
    slog['counts_rw'] = c2d(counts_rw)
    slog['counts_S'] = c2d(counts_S)
    slog['pnc'] = pncs[filename]
    slog['mcc'] = mccs[filename]
    slog['scale_maxes'] = list(scale_maxes)
    slog['ALPHAS'] = list(ALPHAS)
    slog['precision'] = precisions[filename]


    with open(jfile, 'w') as f:
        json.dump(slog, f)

    plt.close('all')

# Post-run Meta-Output and Logging ____________________________________________

timestring = datetime.datetime.now()
timestring = timestring.strftime("%y%m%d_%H%M")

mccfile = os.path.join(OUTPUT_DIR, f"runlog_{timestring}.json")

runlog = {
    'time': timestring,
    'DARK_BG': DARK_BG,
    'DILATE_PER_SCALE': DILATE_PER_SCALE,
    'SCALE_RANGE': SCALE_RANGE,
    'SCALE_TYPE' : SCALE_TYPE,
    'N_SCALES': N_SCALES,
    'scales': list(scales),
    'ALPHAS': list(ALPHAS),
    'BETAS': None,
    'use_npz_files': False,
    'remove_glare': REMOVE_GLARE,
    'files': list(placentas),
    'MCCS': mccs,
    'PNCS': pncs,
    'precisions': precisions
}

# save to a json file
with open(mccfile, 'w') as f:
    json.dump(runlog, f, indent=True)
