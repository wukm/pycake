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

placentas = list_by_quality(0)

beta =0.35
gamma = 0.5
THRESHOLD = .4
scales = np.logspace(-1.5, 3.5, base=2, num=20)

for filename in placentas:
    img = get_named_placenta(filename)
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
    bspine = ecp(img_as_int(spine), disk(3)) > (THRESHOLD*255)

    D = np.ones(img.shape, np.bool)
    D[nf > .05] = 0

    spine_dists = edt(D)
    spine_dists[~thin(bspine)] = 0
    spine_radii = np.round(spine_dists).astype('int')

    dilstack = np.stack([binary_dilation(spine_radii==r, selem=disk(r))
                        for r in range(1,12+1)])

    approx = dilstack.any(axis=0)
    approx_FA = (spine > THRESHOLD)
    approx_PFA = bspine

    m, counts = mcc(approx, trace, bg_mask=img.mask, return_counts=True)
    m_PFA, counts_PFA = mcc(approx_PFA, trace, bg_mask=img.mask, return_counts=True)
    m_FA, counts_FA = mcc(approx_FA, trace, bg_mask=img.mask,
                            return_counts=True)

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])

    p = precision_score(counts)
    p_PFA = precision_score(counts_PFA)
    p_FA = precision_score(counts_FA)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))

    ax[0].imshow(confusion(approx_FA, trace)[crop])
    ax[0].set_title(rf'   fixed $\alpha={THRESHOLD}$', loc='left')
    ax[0].set_title(f'MCC: {m_FA:.2f}\n'
                        f'precision: {p_FA:.2%}', loc='right')

    ax[1].imshow(confusion(bspine,trace)[crop])
    ax[1].set_title(rf'   local percentile $\alpha={THRESHOLD}$', loc='left')
    ax[1].set_title(f'MCC: {m_PFA:.2f}\n'
                        f'precision: {p_PFA:.2%}', loc='right')

    ax[2].imshow(confusion(approx,trace)[crop])
    ax[2].set_title(rf'   local margins$', loc='left')
    ax[2].set_title(f'MCC: {m:.2f}\n'
                        f'precision: {p:.2%}', loc='right')

    [a.axis('off') for a in ax] 

    plt.show()
