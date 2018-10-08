#!/usr/bin/env python3

"""
alpha_sweep_demo.py

show how much variable alphas affect the output.

"""

from get_placenta import get_named_placenta, cropped_args, cropped_view
from get_placenta import list_placentas, open_typefile

from score import compare_trace

from pcsvn import extract_pcsvn, scale_label_figure, apply_threshold
from pcsvn import get_outname_lambda

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

from hfft import fft_gradient
from score import mcc
#filename = 'T-BN0033885.png'
placentas = list_placentas('T-BN')
n_samples = len(placentas)

OUTPUT_DIR = 'output/withdiscrete'
DARK_BG = False
log_range = (-2, 4.5)
n_scales = 20
scales = np.logspace(log_range[0], log_range[1], num=n_scales, base=2)
#alphas = scales**(2/3) / scales[-1]
alphas = [0.1 for s in scales]
#betas = np.linspace(.5, .9, num=n_scales)
betas = None
print(n_samples, "samples total!")
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

    #G = list()

    #for s in scales:
    #    g = fft_gradient(img, s)
    #    G.append(g)
    #G = np.dstack(G)
    #f = F.copy()
    crop = cropped_args(img)
    print("...making outputs")
    outname = get_outname_lambda(filename, output_dir=OUTPUT_DIR)


    approx, labs = apply_threshold(F, alphas, return_labels=True)

    confusion = compare_trace(approx, filename=filename)
    trace = open_typefile(filename, 'trace').astype('bool')
    trace = np.invert(trace)

    m_score, counts = mcc(approx, trace, img.mask, return_counts=True)

    TP, TN, FP, FN = counts

    total = np.invert(img.mask).sum()
    print('TP: {}\t TN: {}\nFP: {}\tFN: {}'.format(TP,TN,FP,FN))
    print('TP+TN+FP+FN={}\ntotal pixels={}'.format(TP+TN+FP+FN,total))

    print("MCC for {}:\t".format(filename), m_score)


    plt.imsave(outname('0_raw'), img[crop].filled(0), cmap=plt.cm.gray)
    plt.imsave(outname('1_confusion'), confusion[crop])
    scale_label_figure(labs, scales, crop=crop, savefilename=outname('2_labeled'),
                       image_only=True)
    plt.imsave(outname('3_fmax'), F.max(axis=-1)[crop],
               vmin=0,vmax=max(alphas),
               cmap=plt.cm.viridis)
    plt.close('all') # something's leaking :(
