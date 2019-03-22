#!/usr/bin/env python3

"""
visual output comparing parametrizations of the Frangi filter. this produces
a 3x3 visual grid of 9 different parametrizations. also outputs

should make this a demonstration as well with a well-chosen inset of a sample
(maybe the same as the frangi scalesweep demos).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, list_placentas)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os
from skimage.morphology import thin
from postprocessing import dilate_to_rim
from scoring import confusion, mcc, integrate_score
from placenta import open_tracefile, open_typefile
from preprocessing import inpaint_hybrid
from placenta import measure_ncs_markings, add_ucip_to_mask

import seaborn as sns
import json

OUTPUT_DIR = (f'demo_output/compare_parametrizations-all-ucip-190322')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

STANDARD = {'label':'standard', 'beta':0.5, 'gamma':0.5}
LOOSE = {'label':'loose', 'beta':1.0, 'gamma':0.30}
STRICT = {'label':'strict', 'beta':0.10, 'gamma':1.0}

ANISOTROPY = {'label':'Anisotropy Factor', 'beta':0.5, 'gamma':0}
SEMISTRICT_BETA = {'label':'semistrict-beta', 'beta':0.10, 'gamma':0.5}
SEMISTRICT_GAMMA = {'label':'semistrict-gamma', 'beta':0.5, 'gamma':1.0}

SEMILOOSE_BETA = {'label':'semiloose-beta', 'beta':1.0, 'gamma':0.5}
SEMILOOSE_GAMMA = {'label':'semiloose-gamma', 'beta':0.5, 'gamma':0.30}
STRUCTURENESS = {'label':'Structureness Factor', 'beta':np.inf, 'gamma':0.5}

#placentas = list_by quality(0)
placentas = list_placentas()

#cm = mpl.cm.plasma
#cm.set_bad('k', 1)  # masked areas are black, not white
cm = mpl.cm.nipy_spectral
cm.set_bad('w', 1)  # masked areas are white, not black

scales = np.logspace(-1.5, 3.2, num=12, base=2)

integral_scores = list()

for filename in placentas:

    cimg = open_typefile(filename, 'raw')
    ctrace = open_typefile(filename, 'ctrace')
    trace = open_tracefile(filename)
    img = get_named_placenta(filename)
    crop = cropped_args(img)
    ucip = open_typefile(filename, 'ucip')
    img = inpaint_hybrid(img)

    # make the size of figures more consistent
    if img[crop].shape[0] > img[crop].shape[1]:
        # and rotating it would be fix all this automatically
        cimg = np.rot90(cimg)
        ctrace = np.rot90(ctrace)
        trace = np.rot90(trace)
        img = np.rot90(img)
        ucip = np.rot90(ucip)
        crop = cropped_args(img)

    ucip_midpoint, resolution = measure_ncs_markings(ucip)
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=50, mask=img.mask)
    old_mask = img.mask.copy()
    img.mask = ucip_mask

    name_stub = filename.rstrip('.png').strip('T-')


    F_demos = list()
    integrals = list()
    PARAMS = [STANDARD, LOOSE, STRICT,
              ANISOTROPY, SEMILOOSE_BETA, SEMISTRICT_BETA,
              STRUCTURENESS, SEMILOOSE_GAMMA, SEMISTRICT_GAMMA]

    for params in PARAMS:
        print(f"running {params['label']} Frangi on {name_stub}")

        if params['gamma'] == 0:
            rescale = False
        else:
            rescale = True

        F_demo = np.stack([frangi_from_image(img, sigma, beta=params['beta'],
                                             gamma=params['gamma'],
                                             dark_bg=False, dilation_radius=20,
                                             rescale_frangi=rescale)
                          for sigma in scales])

        F_max = F_demo.max(axis=0)

        integral = integrate_score(F_max, trace, mask=img.mask)

        F_demos.append(F_max)
        integrals.append(integral)

        del F_demo

    sns.set(font_scale=0.75)
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6.5,7.5))
    A = ax.ravel()


    for i, (Fmax, integral, params) in enumerate(zip(F_demos, integrals,
                                                     PARAMS)):
        beta = params['beta']
        gamma = params['gamma']
        label = params['label']
        A[i].imshow(Fmax[crop], cmap=cm, vmin=0, vmax=1)
        #A[i].imshow(ma.masked_where(img.mask,Fmax)[crop], cmap=cm, vmin=0, vmax=1)
        A[i].set_title(rf'    $\mathcal{{V}}_{{\max}}$ ({label})'+'\n'+
                       rf'    $\beta={beta:.2f}, \gamma={gamma:.2f}$',
                       loc='left')
        A[i].set_title(rf'CVR: {integral:.3f}', loc='right')

    [a.axis('off') for a in A]
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, left=0, right=1, bottom=0,
                        wspace=0, hspace=0.15)
    #fig.tight_layout(h_pad=0.2, w_pad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, ''.join((name_stub,'-3x3params.png')))
                )
    #plt.show()
    plt.close()
    integral_scores.append((name_stub, integrals))

with open(f'{OUTPUT_DIR}/compare_parameters_quality_0.json', 'w') as f:
    json.dump(integral_scores, f, indent=True)

I = [i[1:] for i in integral_scores]
cvrs = np.array(I)
cvrs = cvrs.squeeze()
labels = [p['label'] for p in PARAMS]
I_medians = np.median(I, axis=0)

fig, ax = plt.subplots(figsize=(6.5,7.5))
boxplot_dict = ax.boxplot(cvrs, labels=labels)

axl = plt.setp(ax, xticklabels=labels)
plt.setp(axl, rotation=90)
ax.set_xlabel('Frangi parametrization type')

ax.set_ylabel('Cumulative Vesselness Ratio (CVR)')
ax.set_title(r'Incidence of Vesselness Score along Traced Vessels (all samples)')

# label the median line
for line, med in zip(boxplot_dict['medians'], I_medians.squeeze()):
    x, y = line.get_xydata()[1]  # right of median line
    plt.text(x, y, '%.2f' % med, verticalalignment='center')

plt.subplots_adjust(bottom=0.30)

plt.savefig(f'{OUTPUT_DIR}/cvr_boxplot.png')
plt.show() # maybe adjust size to save manually
