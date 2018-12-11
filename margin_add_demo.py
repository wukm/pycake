#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_int
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

from scipy.ndimage import distance_transform_edt as edt
from skimage.filters.rank import enhance_contrast_percentile as ecp
from skimage.morphology import disk, binary_dilation, thin
from scoring import confusion, mcc

from skimage.exposure import rescale_intensity
from postprocessing import dilate_to_rim
from preprocessing import inpaint_hybrid
import json

placentas = list_by_quality(0)

beta =0.15
gamma = .5
THRESHOLD = .4
scales = np.logspace(-1.5, 3.5, base=2, num=20)

mccs = list()
precs = list()
for filename in placentas:
    print(filename, '*'*30)
    img = get_named_placenta(filename)
    img = inpaint_hybrid(img)
    crop = cropped_args(img)
    trace = open_tracefile(filename)

    F = np.stack([frangi_from_image(img, sigma, beta, gamma,
                                    dark_bg=False, dilation_radius=20,
                                    rescale_frangi=True, signed_frangi=True)
                                    for sigma in scales])

    f = F[:-4].max(axis=0)
    nf = ((-F*(F<0))[:12]).max(axis=0)

    nf = dilate_boundary(nf, mask=img.mask, radius=20).filled(0)
    spine = dilate_boundary(f, mask=img.mask, radius=20).filled(0)
    spine = rescale_intensity(spine, in_range=(0,1), out_range='uint8')
    spine = spine.astype('uint8')
    bspine = ecp(spine, disk(3)) > (THRESHOLD*255)

    approx, radii = dilate_to_rim(bspine, nf > .05, thin_spine=True,
                                  return_radii=True)

    approx2, radii2 = dilate_to_rim(spine > (THRESHOLD*255), nf > .05, thin_spine=True,
                                  return_radii=True)

    approx_FA = (spine > (THRESHOLD*255))
    approx_PFA = bspine

    m, counts = mcc(approx, trace, bg_mask=img.mask, return_counts=True)
    m2, counts2 = mcc(approx2, trace, bg_mask=img.mask, return_counts=True)
    m_PFA, counts_PFA = mcc(approx_PFA, trace, bg_mask=img.mask,
                            return_counts=True)
    m_FA, counts_FA = mcc(approx_FA, trace, bg_mask=img.mask,
                            return_counts=True)

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])

    p = precision_score(counts)
    p2 = precision_score(counts2)
    p_PFA = precision_score(counts_PFA)
    p_FA = precision_score(counts_FA)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25,15))

    ax[0,0].imshow(confusion(approx_FA, trace)[crop])
    ax[0,0].set_title(rf'   fixed $\alpha={THRESHOLD}$', loc='left')
    ax[0,0].set_title(f'MCC: {m_FA:.2f}\n'
                        f'precision: {p_FA:.2%}', loc='right')

    ax[1,0].imshow(confusion(bspine,trace)[crop])
    ax[1,0].set_title(rf'   local percentile $\alpha={THRESHOLD}$', loc='left')
    ax[1,0].set_title(f'MCC: {m_PFA:.2f}\n'
                        f'precision: {p_PFA:.2%}', loc='right')

    ax[1,1].imshow(confusion(approx,trace)[crop])
    ax[1,1].set_title(rf'   dilate_to_margin', loc='left')
    ax[1,1].set_title(f'MCC: {m:.2f}\n'
                        f'precision: {p:.2%}', loc='right')

    ax[1,2].imshow(2*radii[crop], cmap='tab20c')

    ax[0,1].imshow(confusion(approx2,trace)[crop])
    ax[0,1].set_title(rf'   dilate_to_margin (no ecp)', loc='left')
    ax[0,1].set_title(f'MCC: {m2:.2f}\n'
                        f'precision: {p2:.2%}', loc='right')
    im = ax[0,2].imshow(2*radii2[crop], cmap='tab20c')

    #fig.subplots_adjust(right=0.9, wspace=0.05, hspace=0.1)
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax, shrink=0.5)

    [a.axis('off') for a in ax.ravel()]
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.show()

    mccs.append((filename, m_FA, m_PFA, m, m2))
    precs.append((filename, p_FA, p_PFA, p, p2))

runlog = { 'mccs': mccs, 'precs': precs}
with open('181211-margin_add_demo_quality0.json', 'w') as f:
    json.dump(runlog, f, indent=True)
