#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                    mimg_as_float)

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




STRICT = {'label':'strict', 'beta':0.10, 'gamma':1.0}
SEMISTRICT_BETA = {'label':'semistrict-beta', 'beta':0.10, 'gamma':0.5}
SEMISTRICT_GAMMA = {'label':'semistrict-gamma', 'beta':0.5, 'gamma':1.0}

STANDARD = {'label':'standard', 'beta':0.5, 'gamma':0.5}

SEMILOOSE_BETA = {'label':'semiloose-beta', 'beta':1.0, 'gamma':0.5}
SEMILOOSE_GAMMA = {'label':'semiloose-gamma', 'beta':0.5, 'gamma':0.30}
LOOSE = {'label':'loose', 'beta':1.0, 'gamma':0.30}
STRUCTURENESS = {'label':'Structureness Factor', 'beta':np.inf, 'gamma':0.5}
ANISOTROPY = {'label':'Anisotropy Factor', 'beta':0.10, 'gamma':0}
cm = mpl.cm.plasma
#cmscales = mpl.cm.magma
cm.set_bad('k', 1)  # masked areas are black, not white
#cmscales.set_bad('w', 1)

scales = np.logspace(-1.5, 3.5, num=12, base=2)

integral_scores = list()

for filename in list_by_quality(0):

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
    ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=60, mask=img.mask)

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
                                            gamma=params['gamma'], dark_bg=False,
                                            dilation_radius=20,
                                             rescale_frangi=rescale)
                                            for sigma in scales])

        F_max = F_demo.max(axis=0)

        integral = integrate_score(F_max, trace, mask=img.mask)

        F_demos.append(F_max)
        integrals.append(integral)

        del F_demo


    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,20))
    A = ax.ravel()

    #A[0].imshow(cimg[crop])
    #A[0].set_title(name_stub)

    #A[1].imshow(ctrace[crop])
    #A[1].set_title('ground truth')

    for i, (Fmax, integral, params) in enumerate(zip(F_demos, integrals,
                                                     PARAMS)):
        beta = params['beta']
        gamma = params['gamma']
        label = params['label']
        A[i].imshow(ma.masked_where(Fmax==0,Fmax)[crop], cmap=cm, vmin=0, vmax=1)
        A[i].set_title(rf'    $V_{{\max}}$ ({label})'+'\n'+
                       rf'    $\beta={beta:.2f}, \gamma={gamma:.2f}$', loc='left')
        A[i].set_title(rf'CVR: {integral:.3f}', loc='right')

    [a.axis('off') for a in A]
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    integral_scores.append((name_stub, integrals))
