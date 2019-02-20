#!/usr/bin/env python3

"""
this is the main result in which we compare the segmentations across all
samples.

segmentation strategies include:

for a fixed gamma, beta, and range of sigma
1. ISODATA (Frangiless)
2. Frangi simple threshold (high thresh)
3. Frangi simple threshold (low thresh)
4. Frangi nz-percentile (high percentile)
5. Frangi nz-percentile (lower percentile)
6. Margin Add

"""

import numpy as np
import matplotlib.pyplot as plt
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile,
                      strip_ncs_name)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile, apply_threshold
from plate_morphology import dilate_boundary
import os.path, os

from scipy.ndimage import distance_transform_edt as edt
from skimage.filters.rank import enhance_contrast_percentile as ecp
from skimage.morphology import disk, binary_dilation, thin
from scoring import confusion, mcc

from skimage.filters import threshold_isodata
from skimage.exposure import rescale_intensity
from postprocessing import dilate_to_rim
from preprocessing import inpaint_hybrid
import json


def split_signed_frangi_stack(F, negative_range=None, positive_range=None,
                              mask=None):
    """
    F is the frangi stack where the first dimension is the scale space.
    transposing it would be easier.

    if negative_range is given, then only scales within that range are
    accumulated

    will return Vmax(+) and Vmin(-)
    """

    if negative_range is None:
        negative_range = (0, F.shape[-1])

    if positive_range is None:
        positive_range = (0, F.shape[-1])

    f = F[positive_range[0]:positive_range[1]].max(axis=0)
    nf = ((-F*(F<0))[negative_range[0]:negative_range[1]]).max(axis=0)

    if mask is not None:
        nf[mask] = 0

    return f, nf

placentas = list_by_quality(0, N=1)

OUTPUT_DIR = 'output/190220-segmentation_demo'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

beta =0.15
gamma = 1.0
N_scales = 20
THRESHOLD = .3  # \alpha^{(+)}
MARGIN_THRESHOLD = 0.01  # \alpha^{(-)}
NEGATIVE_RANGE = (0,6)
log_range = (-1.5, 3.2)
scales = np.logspace(*log_range, base=2, num=N_scales)

mccs = list()
precs = list()
for filename in placentas:

    # get the name of the sample (like 'BN#######')
    basename = strip_ncs_name(filename)

    print(basename, '*'*30)

    # this calls find_plate_in_raw for all files not in placenta.FAILS
    # in an attempt to improve upon nthe border
    # load sample and do pre-processing
    img = get_named_placenta(filename)
    img = inpaint_hybrid(img)

    # load trace
    crop = cropped_args(img)
    trace = open_tracefile(filename)
    bigmask = dilate_boundary(None, mask=img.mask, radius=20)

    # threshold according to ISODATA threshold for a strawman
    straw = img.filled(0) < threshold_isodata(img.filled(0))
    straw[img.mask] = False  # apply the mask

    # make a MxNxN_scales matrix of Frangi outputs (where (M,N) == img.shape)
    F = np.stack([frangi_from_image(img, sigma, beta, gamma, dark_bg=False,
                                    dilation_radius=20, rescale_frangi=True,
                                    signed_frangi=True).filled(0)
                                    for sigma in scales])

    f, nf = split_signed_frangi_stack(F, positive_range=None,
                                      negative_range=(1,12), mask=bigmask)

    # just in case Vmax(+) and Vmax(-) are both nonzero at the same pixel, we
    # will prioritize Vmax if it is above THRESHOLD, otherwise Vmin if it is
    # above MARGIN threshold, otherwise, Vmax(+)
    # not sure what i really want to show here, maybe just Vmax(+) or just a
    # each over thresholds (i.e. a 3 color map)

    recolor_with_negative = (f <= THRESHOLD) & (nf > MARGIN_THRESHOLD)

    fboth = f.copy()
    fboth[recolor_with_negative] = nf[recolor_with_negative]

    spine = dilate_boundary(f, mask=img.mask, radius=20).filled(0)
    spine = rescale_intensity(spine, in_range=(0,1), out_range='uint8')
    ecp_spine = spine.astype('uint8')
    ecp_spine = ecp(ecp_spine, disk(3)) > (THRESHOLD*255)

    margins = (nf > MARGIN_THRESHOLD)

    # trough filling with ECP prefilter
    approx_td_ecp, radii = dilate_to_rim(ecp_spine, margins, thin_spine=False, return_radii=True)

    approx2, radii2 = dilate_to_rim(spine > THRESHOLD, margins, return_radii=True)

    # fixed threshold
    approx_FA = (spine > THRESHOLD)

    # fixed threshold after ECP prefilter
    approx_PFA = ecp_spine

    # find alphas of each scale (NOTE: THIS DOES *NOT* omit the last 4 scales)
    # scalewise for 95th percentile
    ALPHAS = np.array([nz_percentile(F[k], 95.0)
                       for k in range(len(scales))])
    ALPHAS_98 = np.array([nz_percentile(F[k], 98.0)
                       for k in range(len(scales))])

    approx_PF = apply_threshold(np.transpose(F,(1,2,0)), ALPHAS,
                                return_labels=False)
    approx_PF98 = apply_threshold(np.transpose(F,(1,2,0)), ALPHAS_98,
                                return_labels=False)

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])

    # mcc, counts, precision for ISODATA
    m_st, counts_st = mcc(straw, trace, bg_mask=img.mask, return_counts=True)
    p_st = precision_score(counts_st)

    # mcc, counts, precision f
    #trough_fillings with ECP prefilter
    m_td_ecp, counts_td_ecp = mcc(approx_td_ecp, trace, bg_mask=img.mask, return_counts=True)
    p = precision_score(counts)

    # mcc, counts, precision for trough_fillings w/o ECP prefilter
    m2, counts2 = mcc(approx2, trace, bg_mask=img.mask, return_counts=True)
    p2 = precision_score(counts2)

    # mcc, counts, precision for fixed threshold with ECP prefiler
    m_PFA, counts_PFA = mcc(approx_PFA, trace, bg_mask=img.mask,
                            return_counts=True)
    p_PFA = precision_score(counts_PFA)

    # mcc, counts, precision for fixed threshold w/o prefilter
    m_FA, counts_FA = mcc(approx_FA, trace, bg_mask=img.mask,
                            return_counts=True)
    p_FA = precision_score(counts_FA)

    # mcc, counts, precision for scalewise percent filtelr
    m_PF, counts_PF = mcc(approx_PF, trace, bg_mask=img.mask,
                          return_counts=True)
    p_PF = precision_score(counts_PF)

    m_PF98, counts_PF98 = mcc(approx_PF98, trace, bg_mask=img.mask,
                          return_counts=True)
    p_PF98 = precision_score(counts_PF)

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(25,15))

    ax[0,0].imshow(img[crop], cmap=plt.cm.gray)
    ax[0,0].set_title(basename)

    ax[1,0].imshow(confusion(straw,trace)[crop])
    ax[1,0].set_title(rf'   ISODATA threshold (Frangi-less)', loc='left')
    ax[1,0].set_title(f'MCC: {m_st:.2f}\n'
                        f'precision: {p_st:.2%}', loc='right')

    ax[0,1].imshow(confusion(approx_FA, trace)[crop])
    ax[0,1].set_title(rf'   fixed $\alpha={THRESHOLD}$', loc='left')
    ax[0,1].set_title(f'MCC: {m_FA:.2f}\n'
                        f'precision: {p_FA:.2%}', loc='right')

    #ax[1,1].imshow(confusion(bspine,trace)[crop])
    #ax[1,1].set_title(rf'   local percentile $\alpha={THRESHOLD}$', loc='left')
    #ax[1,1].set_title(f'MCC: {m_PFA:.2f}\n'
    #                    f'precision: {p_PFA:.2%}', loc='right')

    ax[1,1].imshow(confusion(approx_PF,trace)[crop])
    ax[1,1].set_title(rf'   nz-percentile threshold (p=95)', loc='left')
    ax[1,1].set_title(f'MCC: {m_PF:.2f}\n'
                        f'precision: {p_PF:.2%}', loc='right')

    ax[0,2].imshow(confusion(approx2,trace)[crop])
    ax[0,2].set_title(rf'   trough filling', loc='left')
    ax[0,2].set_title(f'MCC: {m2:.2f}\n'
                        f'precision: {p2:.2%}', loc='right')

    ax[1,2].imshow(confusion(approx,trace)[crop])
    ax[1,2].set_title(rf'   trough filling w/ ECP prefilter', loc='left')
    ax[1,2].set_title(f'MCC: {m:.2f}\n'
                        f'precision: {p:.2%}', loc='right')

    ax[0,3].imshow(fboth[crop], vmin=-1.0, vmax=1.0, cmap='seismic')
    ax[0,3].set_title(rf'V_max (signed), $\beta={beta}, \gamma={gamma}$')

    ax[1,3].imshow(trace[crop])
    #fig.subplots_adjust(right=0.9, wspace=0.05, hspace=0.1)
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax, shrink=0.5)

    [a.axis('off') for a in ax.ravel()]
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(os.path.join(OUTPUT_DIR, ''.join(('fig-', basename, '.png'))))
    plt.show()

    mccs.append((filename, m_FA, m_PF, m_PFA, m, m2, m_st))
    precs.append((filename, p_FA, p_PF, p_PFA, p, p2, p_st))

runlog = { 'mccs': mccs, 'precs': precs}

#with open(os.path.join(OUTPUT_DIR,'runlog.json'), 'w') as f:
#    json.dump(runlog, f, indent=True)
