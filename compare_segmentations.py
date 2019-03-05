#!/usr/bin/env python3

"""
this is the main result in which we compare the segmentations across all
samples. rerun this code changing beta & gamma to get outputs for different
segmentations.

segmentation strategies used here are:

for a fixed gamma, beta, and range of sigma
1. ISODATA (Frangiless)
2. Frangi simple threshold (high thresh)
3. Frangi simple threshold (low thresh)
4. Frangi nz-percentile (high percentile)
5. Frangi nz-percentile (lower percentile)
6. trough-filling
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_typefile, open_tracefile,
                      strip_ncs_name, list_placentas, measure_ncs_markings,
                      add_ucip_to_mask)

from frangi import frangi_from_image
from merging import nz_percentile, apply_threshold
from plate_morphology import dilate_boundary
import os.path
import os

# from scipy.ndimage import distance_transform_edt as edt
# from skimage.filters.rank import enhance_contrast_percentile as ecp
from scoring import confusion, mcc

from skimage.color import grey2rgb

from skimage.filters import threshold_isodata
from postprocessing import dilate_to_rim
from preprocessing import inpaint_hybrid
import json


def split_signed_frangi_stack(F, negative_range=None, positive_range=None,
                              mask=None):
    """Get Vmax+ and Vmax- from Frangi stack

    F is the frangi stack where the first dimension is the scale space.
    transposing it would be easier.

    if negative_range is given, then only scales within that range are
    accumulated

    will return Vmax(+) and Vmin(-)

    should redo this so that it returns V_{\Sigma}^{(+)} and V_{\Sigma}^{(-)}
    """

    if negative_range is None:
        negative_range = (0, F.shape[-1])

    if positive_range is None:
        positive_range = (0, F.shape[-1])

    f = F[positive_range[0]:positive_range[1]].max(axis=0)
    nf = ((-F*(F < 0))[negative_range[0]:negative_range[1]]).max(axis=0)

    if mask is not None:
        nf[mask] = 0

    return f, nf


#quality_name = 'good'
#placentas = list_by_quality(0, N=2)

quality_name = 'all'
placentas = list_placentas()


#beta, gamma, parametrization_name = 0.15, 1.0, "strict"
#beta, gamma, parametrization_name = 0.15, 0.5, "semistrict"
#beta, gamma, parametrization_name = 0.5, 1.0, 'semistrict-gamma'
beta, gamma, parametrization_name = 0.5, 0.5, "standard"

OUTPUT_DIR = (f'output/190304-segmentation_demo_'
              f'{quality_name}_{parametrization_name}-ucip')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N_scales = 20
THRESHOLD = .3  # \alpha^{(+)}
THRESHOLD_LOW = .2  # \alpha^{(+)}
MARGIN_THRESHOLD = 0.01  # \alpha^{(-)}
NEGATIVE_RANGE = (0, 6)
log_range = (-1, 3.2)
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
    ucip, res = measure_ncs_markings(filename=filename)
    old_mask = img.mask.copy()
    img.mask = add_ucip_to_mask(ucip, radius=50, mask=img.mask)

    # load trace
    crop = cropped_args(img)
    cimg = open_typefile(filename, 'raw')
    ctrace = open_typefile(filename, 'ctrace')
    trace = open_tracefile(filename)
    bigmask = dilate_boundary(None, mask=img.mask, radius=20)

    # rotate the image so it's hotdog not hamburger
    if img[crop].shape[0] > img[crop].shape[1]:
        # and rotating it would be fix all this automatically
        cimg = np.rot90(cimg)
        ctrace = np.rot90(ctrace)
        trace = np.rot90(trace)
        img = np.rot90(img)
        crop = cropped_args(img)
        bigmask = dilate_boundary(None, mask=img.mask, radius=20)
        old_mask = np.rot90(old_mask)

    # threshold according to ISODATA threshold for a strawman
    straw = img.filled(0) < threshold_isodata(img.filled(0))
    straw[img.mask] = False  # apply the mask

    # make a MxNxN_scales matrix of Frangi outputs (where (M,N) == img.shape)
    F = np.stack([frangi_from_image(img, sigma, beta, gamma, dark_bg=False,
                                    dilation_radius=20, rescale_frangi=True,
                                    signed_frangi=True).filled(0)
                 for sigma in scales])

    # Fneg still has negative direction but these are now separated
    Fpos = np.clip(F, 0, None)  # replace negative values with 0
    Fneg = np.clip(F, None, 0)  # replace positive values with 0

    # makes Vmaxpos and Vmaxneg (could also take max of Fpos and -Fneg above
    f, nf = split_signed_frangi_stack(F, positive_range=None,
                                      negative_range=NEGATIVE_RANGE,
                                      mask=bigmask)

    # everything in the following section needs a better name :/
    spine = dilate_boundary(f, mask=img.mask, radius=20).filled(0)
    spineseeds = (spine > THRESHOLD)
    margins = (nf > MARGIN_THRESHOLD)

    # trough-filling segmentation
    approx_tf, radii = dilate_to_rim(spineseeds > THRESHOLD, margins,
                                     return_radii=True)

    # fixed threshold
    approx_FA_high = (spine > THRESHOLD)
    approx_FA_low = (spine > THRESHOLD_LOW)

    # scalewise for 95th percentile
    ALPHAS_95 = np.array([nz_percentile(Fpos[k], 95.0)
                       for k in range(len(scales))])
    ALPHAS_98 = np.array([nz_percentile(Fpos[k], 98.0)
                       for k in range(len(scales))])

    approx_PF95 = apply_threshold(np.transpose(Fpos,(1,2,0)), ALPHAS_95,
                                return_labels=False)
    approx_PF98 = apply_threshold(np.transpose(Fpos,(1,2,0)), ALPHAS_98,
                                return_labels=False)

    precision_score = lambda t: int(t[0]) / int(t[0] + t[2])

    # mcc, counts, precision for ISODATA
    m_st, counts_st = mcc(straw, trace, bg_mask=img.mask, return_counts=True)
    p_st = precision_score(counts_st)

    # mcc, counts, precision for fixed threshold w/o prefilter
    m_FA_low, counts_FA_low = mcc(approx_FA_low, trace, bg_mask=img.mask,
                            return_counts=True)
    p_FA_low = precision_score(counts_FA_low)

    m_FA_high, counts_FA_high = mcc(approx_FA_high, trace, bg_mask=img.mask,
                            return_counts=True)
    p_FA_high = precision_score(counts_FA_high)

    # mcc, counts, precision for scalewise percent filtelr
    m_PF95, counts_PF95 = mcc(approx_PF95, trace, bg_mask=img.mask,
                          return_counts=True)
    p_PF95 = precision_score(counts_PF95)

    m_PF98, counts_PF98 = mcc(approx_PF98, trace, bg_mask=img.mask,
                          return_counts=True)
    p_PF98 = precision_score(counts_PF98)

    #mcc, counts, precision for trough_fillings w/o ECP prefilter
    m_tf, counts_tf = mcc(approx_tf, trace, bg_mask=img.mask, return_counts=True)
    p_tf = precision_score(counts_tf)




    # this is just used for visualization, not very meaningful though
    #recolor_with_negative = (f <= THRESHOLD) & (nf > MARGIN_THRESHOLD)
    #fboth = f.copy()
    #fboth[recolor_with_negative] = nf[recolor_with_negative]

    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(13,7))

    I = grey2rgb(img.data)  # three channels (based on img old data)
    I[old_mask] = (1.,1.,1.)  # make old BG white not black
    I[img.mask] *= (1.0, 0.8, 0.8)  # overlay red on ucip mask
    #ax[0,0].imshow(img[crop], cmap=plt.cm.gray)
    ax[0,0].imshow(I[crop])
    ax[0,0].set_title(basename)


    ax[0,1].imshow(confusion(approx_FA_high, trace)[crop])
    ax[0,1].set_title(rf'   fixed $\alpha={THRESHOLD}$', loc='left')
    ax[0,1].set_title(f'MCC: {m_FA_high:.2f}\n'
                        f'precision: {p_FA_high:.2%}', loc='right')

    ax[0,2].imshow(confusion(approx_FA_low, trace)[crop])
    ax[0,2].set_title(rf'   fixed $\alpha={THRESHOLD_LOW}$', loc='left')
    ax[0,2].set_title(f'MCC: {m_FA_low:.2f}\n'
                        f'precision: {p_FA_low:.2%}', loc='right')

    ax[0,3].imshow(confusion(straw,trace)[crop])
    ax[0,3].set_title(f'   ISODATA\n   (Frangi-less)', loc='left')
    ax[0,3].set_title(f'MCC: {m_st:.2f}\n'
                        f'precision: {p_st:.2%}', loc='right')

    ax[1,0].imshow(f[crop], cmap='nipy_spectral', vmax=1.0, vmin=0.0)
    ax[1,0].set_title(r'$\mathcal{V}_{\max}$  '
                      fr'$\beta={beta}, \gamma={gamma}$')

    ax[1,1].imshow(confusion(approx_PF95,trace)[crop])
    ax[1,1].set_title(f'   scalewise nz-p\n   (p=95)', loc='left')
    ax[1,1].set_title(f'MCC: {m_PF95:.2f}\n'
                        f'precision: {p_PF95:.2%}', loc='right')

    ax[1,2].imshow(confusion(approx_PF98,trace)[crop])
    ax[1,2].set_title(f'   scalewise nz-p\n   (p=98)', loc='left')
    ax[1,2].set_title(f'MCC: {m_PF98:.2f}\n'
                        f'precision: {p_PF98:.2%}', loc='right')
    ax[1,3].imshow(confusion(approx_tf,trace)[crop])
    ax[1,3].set_title( '   trough-fill\n'
                      r'    $\alpha^{(+)}=$'
                      fr'${THRESHOLD}$', loc='left')
    ax[1,3].set_title(f'MCC: {m_tf:.2f}\n'
                        f'precision: {p_tf:.2%}', loc='right')




    [a.axis('off') for a in ax.ravel()]
    fig.tight_layout()
    fig.subplots_adjust(right=1.0, left=0, top=0.95, bottom=0.,
                        wspace=0.0, hspace=0.05)
    plt.savefig(os.path.join(OUTPUT_DIR, ''.join(('fig-', basename, '.png'))))
    #plt.show()
    plt.close()

    mccs.append((filename, m_FA_high, m_FA_low, m_PF95, m_PF98, m_st, m_tf))
    precs.append((filename, p_FA_high, p_FA_low, p_PF95, p_PF98, p_st, p_tf))

runlog = { 'mccs': mccs, 'precs': precs}

with open(os.path.join(OUTPUT_DIR,'runlog.json'), 'w') as f:
    json.dump(runlog, f, indent=True)

# get rid of sample labels
M = np.array([m[1:] for m in mccs])
P = np.array([p[1:] for p in precs])

M_medians = np.median(M, axis=0)  # what the actual medians are (for labeling)
P_medians = np.median(P, axis=0)  # what the actual medians are (for labeling)

# segmentation strategy labels
labels = [
    rf'thresh-high $\alpha={THRESHOLD}$',
    rf'thresh-low $\alpha={THRESHOLD_LOW}$',
    'snz-p\n(p=95)',
    'snz-p\n(p=98)',
    'ISODATA',
    'trough-fill'
]

# make a bunch of boxplots
for scorename, data, medians in [('MCC', M, M_medians),
                                 ('precision', P, P_medians)]:

    # would like to combine these into the same plot
    # easier to do with xarray, maybe just do processing in a different script
    fig, ax = plt.subplots()
    boxplot_dict = ax.boxplot(data, labels=labels)
    axl = plt.setp(ax, xticklabels=labels)
    plt.setp(axl, rotation=90)
    ax.set_xlabel('segmentation method')
    ax.set_title(f'{scorename} scores of segmentation methods'
                 f'({quality_name} samples),'
                 f'{parametrization_name} parametrization')
    ax.set_ylabel(scorename)

    # label medians, from https://stackoverflow.com/a/18861734
    for line, med in zip(boxplot_dict['medians'], medians):
        x, y = line.get_xydata()[1] # right of median line
        plt.text(x, y, '%.2f' % med, verticalalignment='center')

    # you have to manually prevent clipping of rotated labels, amazing
    plt.subplots_adjust(bottom=0.30)
    plt.tight_layout()
    boxplot_name = '-'.join((quality_name, scorename, "boxplot",
                             parametrization_name))
    plt.savefig(os.path.join(OUTPUT_DIR, boxplot_name + '.png'))
    plt.show()
    plt.close()
