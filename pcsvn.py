#!/usr/bin/env python3

from placenta import get_named_placenta
from hfft import fft_hessian
from diffgeo import principal_curvatures, principal_directions
from frangi import get_frangi_targets
from skimage.util import img_as_float
import numpy as np
from preprocessing import (inpaint_glare, inpaint_with_boundary_median,
                           inpaint_hybrid)

from plate_morphology import dilate_boundary

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.ma as ma

import os.path
import json
import datetime


def make_multiscale(img, scales, betas, gammas,
                    find_principal_directions=False, dilate_per_scale=True,
                    signed_frangi=False, dark_bg=True, kernel=None,
                    VERBOSE=True):
    """Returns an ordered list of dictionaries for each scale of Frangi info.

    Each element in the output contains the following info:
        {'sigma': sigma,
         'beta': beta,
         'gamma': gamma,
         'H': hesh,
         'F': targets,
         'k1': k1,
         'k2': k2,
         't1': t1, # if find_principal_directions
         't2': t2 # if find_principal_directions
         }

    """

    # store results of each scale (create as empty list)
    multiscale = list()

    img = ma.masked_array(img_as_float(img), mask=img.mask)

    for i, sigma, beta, gamma in zip(range(len(scales)), scales,
                                     betas, gammas):
        if dilate_per_scale:
            if sigma < 2.5:
                radius = 10
            elif sigma > 20:
                radius = int(sigma*2)
            else:
                radius = int(sigma*4)  # a little aggressive
        else:
            radius = None

        if VERBOSE:
            print('σ={}'.format(sigma))

        # get hessian components at each pixel as a triplet (Lxx, Lxy, Lyy)
        hesh = fft_hessian(img, sigma, kernel=kernel)

        if VERBOSE:
            print('finding principal curvatures')

        # calculate principal curvatures with |k1| <= |k2|
        k1, k2 = principal_curvatures(img, sigma=sigma, H=hesh)

        # area of influence to zero out
        if dilate_per_scale:
            collar = dilate_boundary(None, radius=radius, mask=img.mask)

            k1[collar] = 0
            k2[collar] = 0

        # set anisotropy parameter if not specified
        if gamma is None:
            # Frangi suggested 'half the max Hessian norm' as an empirical
            # half the max spectral radius is easier to calculate so do that
            # shouldn't be affected by mask data but should make sure the
            # mask is *well* far away from perimeter
            # we actually calculate half of max hessian norm
            # using frob norm = sqrt(trace(AA^T))
            hxx, hxy, hyy = hesh
            hessian_norm = np.sqrt((hxx**2 + 2*hxy**2 + hyy**2))

            # make sure the max doesn't occur on a boundary
            dilation_radius = int(max(np.ceil(sigma), 10))
            collar = dilate_boundary(None, radius=dilation_radius,
                                     mask=img.mask)
            hessian_norm[collar] = 0
            max_hessian_norm = hessian_norm.max()
            gamma = .5*max_hessian_norm

            if VERBOSE:
                # compare to other method of calculating gamma
                gamma_alt = .5 * np.abs(k2).max()
                print(f"half of k2 max is {gamma_alt}")

        if VERBOSE:
            print(f"gamma (half of max hessian (frob) norm is {gamma}")
            print(f'finding Frangi targets with β={beta} and γ={gamma:.2}')

        # calculate frangi targets at this scale
        targets = get_frangi_targets(k1, k2, beta=beta, gamma=gamma,
                                     dark_bg=dark_bg, signed=signed_frangi)


        # store results as a dictionary
        this_scale = {'sigma': sigma,
                      'beta': beta,
                      'gamma': gamma,
                      'H': hesh,
                      'F': targets,
                      'k1': k1,
                      'k2': k2,
                      'border_radius': radius
                      }

        if find_principal_directions:
            # principal directions should only be computed for critical regions
            # ignore anything less than a std deviation over the mean
            # this mask is where PD's will *NOT* be calculated
            cutoff = targets.mean() + targets.std()
            pd_mask = np.bitwise_or(targets < cutoff, img.mask).filled(1)
            percent_calculated = (pd_mask.size - pd_mask.sum()) / pd_mask.size

            if VERBOSE:
                print(f"finding PD's for {percent_calculated:.2%} of image"
                      f"anything above vesselness score {cutoff:.6f}"
                      )
            t1, t2 = principal_directions(img, sigma=sigma, H=hesh,
                                          mask=pd_mask)

            # add them to this scale's output
            this_scale['t1'] = t1
            this_scale['t2'] = t2
        else:
            if VERBOSE:
                print('skipping principal direction calculation')

        # store results as a list of dictionaries
        multiscale.append(this_scale)

    return multiscale


def extract_pcsvn(filename, scales, alphas=None, betas=None, gammas=None,
                  DARK_BG=True, dilate_per_scale=True, verbose=True,
                  generate_json=True, output_dir=None, kernel=None,
                  signed_frangi=False, remove_glare=False):
    """Run PCSVN extraction on the sample given in the file.

    Despite the name, this simply returns the Frangi filter responses at
    each provided scale without explicitly making any decisions about what
    is or is not part of the PCSVN.

    TODO:Finish docstring!
    """

    raw_img = get_named_placenta(filename, maskfile=None)

    # Multiscale & Frangi Parameters######################

    # set default alphas and betas if undeclared
    if alphas is None:
        alphas = [.15 for s in scales]  # threshold constant
    if betas is None:
        betas = [0.5 for s in scales]  # anisotropy constant

    # declare None here to calculate half of hessian's norm
    if gammas is None:
        gammas = [None for s in scales]  # structureness parameter

    # Preprocessing###############
    if remove_glare:
        if verbose:
            print('removing glare from sample')
        # img = inpaint_glare(raw_img)
        # img = inpaint_with_boundary_median(raw_img)
        img = inpaint_hybrid(raw_img)
    else:
        img = raw_img.copy()  # in case we alter the mask or something

    # Multiscale Frangi Filter##############################

    # output is a dictionary of relevant info at each scale
    multiscale = make_multiscale(img, scales, betas, gammas,
                                 find_principal_directions=False,
                                 dilate_per_scale=dilate_per_scale,
                                 kernel=kernel,
                                 signed_frangi=signed_frangi,
                                dark_bg=DARK_BG,
                                 VERBOSE=verbose)

    # extract these for logging
    gammas = [scale['gamma'] for scale in multiscale]
    border_radii = [scale['border_radius'] for scale in multiscale]

    ###Process Multiscale Targets############################

    # ignore targets too close to edge of plate
    # wait are we doing this twice?
    if dilate_per_scale:
        if verbose:
            print('trimming collars of plates (per scale)')

        for i in range(len(multiscale)):
            f = multiscale[i]['F']
            # twice the buffer (be conservative!)
            radius = int(multiscale[i]['sigma']*2)
            if verbose:
                print('dilating plate for radius={}'.format(radius))
            f = dilate_boundary(f, radius=radius, mask=img.mask)
            # get rid of mask
            multiscale[i]['F'] = f.filled(0)
    else:
        for i in range(len(multiscale)):
            # get rid of mask
            multiscale[i]['F'] = multiscale[i]['F'].filled(0)

    ###Make Composite#########################################

    # get a M x N x n_scales array of Frangi targets at each level
    F_all = np.dstack([scale['F'] for scale in multiscale])

    if generate_json:

        time_of_run = datetime.datetime.now()
        timestring = time_of_run.strftime("%y%m%d_%H%M")

        logdata = {'time': timestring,
                'filename': filename,
                'alphas': list(alphas),
                'betas': list(betas),
                'gammas': gammas,
                'sigmas': list(scales),
                }
        if dilate_per_scale:
            logdata['border_radii'] = border_radii

        if output_dir is None:
            output_dir = 'output'

        base = os.path.basename(filename)
        *base, suffix = base.split('.')
        dumpfile = os.path.join(output_dir,
                                ''.join(base) + '_' + str(timestring)
                                + '.json')

        with open(dumpfile, 'w') as f:
            json.dump(logdata, f, indent=True)

    # this function used to returns scales and alphas too but now doesn't,
    return F_all, img


def get_outname_lambda(filename, output_dir=None, timestring=None):
    """
    return a lambda function which can build output filenames
    """

    if output_dir is None:
        output_dir = 'output'

    base = os.path.basename(filename)
    *base, suffix = base.split('.')

    if timestring is None:
        time_of_run = datetime.datetime.now()
        timestring = time_of_run.strftime("%y%m%d_%H%M")

    outputstub = ''.join(base) +'_' + timestring +  '_{}.'+ suffix
    return lambda s: os.path.join(output_dir, outputstub.format(s))


def _build_scale_colormap(N_scales, base_colormap, basecolor=(0,0,0,1)):
    """
    returns a mpl.colors.ListedColormap with N samples,
    based on the colormap named "default_colormap" (a string)

    the N colors are given by the default colormap, and
    basecolor (default black) is added to map to 0.
    (you could change this, for example, to (1,1,1,1) for white)

    reversed colormaps often work better if the basecolor is black
    you should make sure there's good contrast between the basecolor
    and the first color in the colormap
    """

    map_range = np.linspace(0, 1, num=N_scales)

    colormap = plt.get_cmap(base_colormap)

    colorlist = colormap(map_range)

    # add basecolor as the first entry
    colorlist = np.vstack((basecolor, colorlist))

    return mpl.colors.ListedColormap(colorlist)

def scale_label_figure(wheres, scales, savefilename=None,
                       crop=None, show_only=False, image_only=False,
                       save_colorbar_separate=False, savecolorbarfile=None,
                       output_dir=None):
    """
    crop is a slice object.
    if show_only, then just plt.show (interactive).
    if image_only, then this will *not* be printed with the colorbar

    if save_colormap_separate, then the colormap will be saved as a separate
    file
    """
    if crop is not None:
        wheres = wheres[crop]

    fig, ax = plt.subplots() # not sure about figsize
    N = len(scales) # number of scales / labels

    tabemap = _build_scale_colormap(N, 'viridis_r')

    if image_only:
        plt.imsave(savefilename, wheres, cmap=tabemap, vmin=0, vmax=N)
        plt.close()
    else:
        imgplot = ax.imshow(wheres, cmap=tabemap, vmin=0, vmax=N)
        # discrete colorbar
        cbar = plt.colorbar(imgplot)

        # this is apparently hackish, beats me
        tick_locs = (np.arange(N+1) + 0.5)*(N-1)/N

        cbar.set_ticks(tick_locs)
        # label each tick with the sigma value
        scalelabels = [r"$\sigma = {:.2f}$".format(s) for s in scales]
        scalelabels.insert(0, "(no match)")
        # label with their sigma value
        cbar.set_ticklabels(scalelabels)
        #ax.set_title(r"Scale ($\sigma$) of maximum vesselness ")
        plt.tight_layout()

        #plt.savefig(outname('labeled'), dpi=300)
        if show_only or (savefilename is None):
            plt.show()
        else:
            plt.savefig(savefilename, dpi=300)

        plt.close()

    if save_colorbar_separate:
        if savecolorbarfile is None:
            savecolorbarfile = os.path.join(output_dir, "scale_colorbar.png")
        fig = plt.figure(figsize=(1, 8))
        ax1 = fig.add_axes([0.05, 0.05, 0.15, 0.9])
        tick_locs = (np.arange(N+1) + 0.5)*(N-1)/N
        scalelabels = [r"$\sigma = {:.2f}$".format(s) for s in scales]
        scalelabels.insert(0, "n/a")
        cbar = mpl.colorbar.ColorbarBase(ax1, cmap=tabemap,
                                         norm=mpl.colors.Normalize(vmin=0,
                                                                   vmax=N),
                                         orientation='vertical',
                                         ticks=tick_locs)
        cbar.set_ticklabels(scalelabels)
        plt.savefig(savecolorbarfile, dpi=300)
