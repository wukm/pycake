#!/usr/bin/env python3

"""
This is the main program. It approximates the PCSVN


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

# for some post_processing, this needs to be moved elsewhere
from skimage.filters import sobel
from frangi import frangi_from_image
from plate_morphology import dilate_boundary
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import random_walker



# INITIALIZE SAMPLES ________________________________________________________

# initialize a list of samples (several different ways)
#placentas = list_by_quality(0)
placentas = list_placentas('T-BN')  # load allllll placentas
#placentas = list_by_quality(0)
#placentas = list_by_quality(json_file='manual_batch.json')
#placentas = ['T-BN0204423.png'] # for a single sample, use a 1 element list.

n_samples = len(placentas)

# RUNTIME OPTIONS ___________________________________________________________

MAKE_NPZ_FILES = False # pickle frangi targets if you can
USE_NPZ_FILES = False  # use old npz files if you can
NPZ_DIR = 'output/181120-smaller_bounds' # where to look for npz files
OUTPUT_DIR = 'output/181120-smaller_bounds' # where to save outputs

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

# use preprocessing.inpaint_with_boundary_median() to replace high
# glare regions
REMOVE_GLARE = True

log_range = (-2, 3.5)
n_scales = 40

# when showing "large scales only", this is where to start
# (some index between 0 and n_scales)
LO_offset = 8


scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
alphas = [0.15 for s in scales]
betas = None  # will be given default parameters
gammas =  None  # will be given default parameters

mccs = dict()  # empty dict to store MCC's of each sample
pncs = dict() # empty dict to store percent network covered for each sample

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
    else:
        npz_filename = None

    # set a lambda function to make output file names
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)

    raw_img = get_named_placenta(filename, maskfile=None)
    if npz_filename is not None:
        F = np.load(npz_filename)['F']
        if REMOVE_GLARE:
            img = inpaint_hybrid(raw_img)
        else:
            img = raw_img
        print('successfully loaded the frangi targets!')

    else:
        print('finding multiscale frangi targets')

        F, img = extract_pcsvn(filename, DARK_BG=DARK_BG, alphas=alphas,
                            betas=betas, scales=scales, gammas=gammas,
                            kernel='discrete', dilate_per_scale=True,
                            verbose=False, signed_frangi=SIGNED_FRANGI,
                            generate_json=True, output_dir=OUTPUT_DIR,
                            remove_glare=REMOVE_GLARE)

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

    Fmax = F.max(axis=-1)

    # print(f"The umbilicial cord insertion point is at {ucip_midpoint}")
    # print(f"The resolution of the image is {resolution} pixels per cm.")

    # mask anywhere close to (within 90px L2 distance of) the UCIP
    # this is imperically how the it is, although could be bigger
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=90, mask=img.mask)

    # open up RGB tracefiles (keep for visualizing?)
    A_trace = open_typefile(filename, 'arteries')
    V_trace = open_typefile(filename, 'veins')

    skeltrace = skeletonize_trace(A_trace, V_trace)

    # matrix of widths of traced image
    widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')
    # min_widths = merge_widths_from_traces(A_trace, V_trace,
    #                                       strategy='minimum')

    # trace ignoring largest vessels (19 pixels wide)
    # trace_smaller_only = filter_widths(min_widths, min_width=3, max_width=17)
    # trace_smaller_only != 0
    # use limited scales
    approx_LO, labs_LO = apply_threshold(F[:,:, LO_offset:], alphas[LO_offset:])

    # fix labels to incorporate offset
    labs_LO = (labs_LO != 0)*(labs_LO + LO_offset)

    # confusion matrix against default trace
    confuse = confusion(approx, trace, bg_mask=ucip_mask)
    confuse_LO = confusion(approx_LO, trace,
                                    bg_mask=ucip_mask)

    m_score, counts = mcc(approx, trace, ucip_mask, return_counts=True)
    m_score_LO, counts_LO = mcc(approx_LO, trace, ucip_mask,
                                return_counts=True)

    # this all just verifies that the 4 categories were added up
    # correctly and match the total number of pixels in the reported
    # placental plate.
    TP, TN, FP, FN = counts # return these for more analysis?

    total = np.invert(ucip_mask).sum()
    print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    print('TP+TN+FP+FN={}\ttotal pixels={}'.format(TP+TN+FP+FN,total))

    # MOVE THIS ELSEWHERE
    s = sobel(img)
    s = dilate_boundary(s, mask=img.mask, radius=20)
    finv = frangi_from_image(s, sigma=0.8, dark_bg=True)
    finv_thresh = nz_percentile(finv, 80)
    margins = remove_small_objects((finv > finv_thresh).filled(0), min_size=32)
    margins_added = np.logical_or(margins, approx)
    margins_added = remove_small_holes(margins_added, min_size=100,
                                       connectivity=2)
    # random walker markers
    markers = np.zeros(img.shape, dtype=np.uint8)
    markers[Fmax < .1] = 1
    markers[margins_added] = 2
    rw = random_walker(img, markers, beta=1000)
    approx_rw = (rw==2)
    confuse_rw = confusion(approx_rw, trace, bg_mask=ucip_mask)
    m_score_rw = mcc(approx_rw, trace, ucip_mask)
    pnc_rw = np.logical_and(skeltrace, approx_rw).sum() / skeltrace.sum()

    mccs[filename] =  (m_score, m_score_LO, m_score_rw)

    print(f'mcc score of {m_score:.3} for {filename}')
    print(f'mcc score of {m_score_LO:.3} with larger sigmas only')
    print(f'mcc score of {m_score_rw:.3} after random walker')

    plt.imsave(outname('0_raw'), raw_img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('1_img'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('4_confusion'), confuse[crop])
    plt.imsave(outname('7_confusion_LO'), confuse_LO[crop])
    plt.imsave(outname('9_confusion_rw'), confuse_rw[crop])

    percent_covered = np.logical_and(skeltrace, approx).sum() / skeltrace.sum()
    percent_covered_LO = np.logical_and(skeltrace, approx_LO).sum() / skeltrace.sum()

    pncs[filename] = (percent_covered, percent_covered_LO, pnc_rw)

    print('percentage of skeltrace covered:', f'{percent_covered:.2%}')
    print('percentage of skeltrace covered (larger sigmas only):',
          f'{percent_covered_LO:.2%}')
    print('percentage of skeltrace covered (random_walker):',
          f'{pnc_rw:.2%}')
    plt.imsave(outname('5_coverage'), confusion(approx, skeltrace)[crop])
    plt.imsave(outname('8_coverage_LO'), confusion(approx_LO, skeltrace)[crop])
    plt.imsave(outname('9_coverage_rw'), confusion(approx_rw, skeltrace)[crop])

    # only save the colorbar the first time
    save_colorbar = (i==0)
    # make the graph that shows what scale the max was pulled from
    scale_label_figure(labs, scales, crop=crop,
                       savefilename=outname('3_labeled'), image_only=True,
                       save_colorbar_separate=save_colorbar,
                       output_dir=OUTPUT_DIR)

    scale_label_figure(labs_LO, scales, crop=crop,
                       savefilename=outname('4_labeled'), image_only=True,
                       save_colorbar_separate=False, output_dir=OUTPUT_DIR)
    # save the maximum frangi output
    plt.imsave(outname('2_fmax'), F.max(axis=-1)[crop],
               vmin=0, vmax=1.0, cmap=plt.cm.nipy_spectral)
    plt.close('all')  # something's leaking :(

# json file with mccs and other runtime info
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

with open(mccfile, 'w') as f:
    json.dump(runlog, f, indent=True)
