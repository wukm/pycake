#!/usr/bin/env python3

from get_placenta import get_named_placenta
from score import compare_trace
from hfft import fft_hessian
from diffgeo import principal_curvatures, principal_directions
from frangi import get_frangi_targets
import numpy as np
import numpy.ma as ma

from skimage.morphology import label, skeletonize

from plate_morphology import dilate_boundary

import matplotlib.pyplot as plt
import matplotlib as mpl

import os.path
import json
import datetime

from get_placenta import cropped_args

def make_multiscale(img, scales, betas, gammas, find_principal_directions=False,
                    dilate_per_scale=True, signed_frangi=False,
                    dark_bg=True, kernel=None, VERBOSE=True):
    """
    returns an ordered list of dictionaries for each scale
    multiscale.append(
        {'sigma': sigma,
         'beta': beta,
         'gamma': gamma,
         'H': hesh,
         'F': targets,
         'k1': k1,
         'k2': k2,
         't1': t1,
         't2': t2
         }
    """

    # store results of each scale (create as empty list)
    multiscale = list()

    img = img / 255.

    for i, sigma, beta, gamma in zip(range(len(scales)), scales, betas, gammas):
        if dilate_per_scale:
            if sigma < 2.5:
                radius = 10
            elif sigma >  20:
                radius = int(sigma*2)
            else:
                radius = int(sigma*4) # a little aggressive
        else:
            radius = None

        if VERBOSE:
            print('σ={}'.format(sigma))

        # get hessian components at each pixel as a triplet (Lxx, Lxy, Lyy)
        hesh = fft_hessian(img, sigma, kernel=kernel)

        if VERBOSE:
            print('finding principal curvatures')


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
            #gamma_alt = .5 * np.abs(k2).max()
            #print("half of k2 max is", gamma_alt)

            # or actually calculate half of max hessian norm
            # using frob norm = sqrt(trace(AA^T))
            hxx, hxy, hyy = hesh
            hessian_norm = np.sqrt((hxx**2 + 2*hxy**2 + hyy**2))

            # make sure the max doesn't occur on a boundary
            # this is rough and bad and could change a lot.
            dilation_radius = int(max(np.ceil(sigma), 10))
            collar = dilate_boundary(None, radius=dilation_radius,
                                     mask=img.mask)
            hessian_norm[collar] = 0
            max_hessian_norm = hessian_norm.max()
            gamma = .5*max_hessian_norm

            #print("gamma (half of max hessian norm is)", gamma)
        if VERBOSE:
            print('finding Frangi targets with β={} and γ={:.2}'.format(beta, gamma))

        # calculate frangi targets at this scale
        targets = get_frangi_targets(k1,k2,
                    beta=beta, gamma=gamma, dark_bg=dark_bg,
                                     signed=signed_frangi, threshold=False)

        #store results as a dictionary
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
            # principal directions will only be computed for significant regions
            pd_mask = np.bitwise_or(targets < (targets.mean() + targets.std()),
                                    img.mask).filled(1)

            if VERBOSE:
                percentage_calculated = (pd_mask.size - pd_mask.sum()) / pd_mask.size
                print('finding principal directions for {:.2%} of the image'.format(percentage_calculated))

            t1, t2 = principal_directions(img, sigma=sigma, H=hesh, mask=pd_mask)

            this_scale['t1'] = t1
            this_scale['t2'] = t2
        else:
            if VERBOSE:
                print('skipping principal direction calculation')

        # store results as a dictionary
        multiscale.append(this_scale)

    return multiscale



def apply_threshold(targets, alphas, return_labels=True):
    """
    if return_labels is True, return 1,..,n for scale at which
    max target was found (or 0 if no match),
    otherwise simply return a binary matrix

    targets is a (M,N,n) shape matrix (like F_all)
    alphas is a list / 1d array of alphas of length n


    if alphas is a single number, then this should work just fine

    for convenience,

    if return labels is true, this will return both the final
    threshold and the labels as two separate matrices
    """

    # you could make this work in 2D if you wanted to so alphas is a
    # constant and targets is only 2D but that's a later day

    # make it an array (even if it's a single element)
    alphas = np.array(alphas)

    # if it's just a MxN matrix, expand it trivially so it works below

    if targets.ndim == 2:

        targets = np.expand_dims(targets,2)

    # either there's an alpha for each channel or there's a single
    # alpha to be broadcast across all channels
    assert (targets.shape[-1] == alphas.size) or (alphas.size == 1)

    # pixels that passed the threshold at any level
    passed = (targets >  alphas).any(axis=-1)

    if not return_labels:

        # works by broadcasting
        return passed

    # get label of where maximum occurs
    wheres = targets.argmax(axis=-1)

    # reserve 0 label for no match
    wheres += 1

    # then remove anything that didn't pass the threshold
    wheres[np.invert(passed)] = 0

    assert np.all( passed == (wheres > 0) )

    return passed, wheres

def extract_pcsvn(filename, scales,
                  alphas=None, betas=None, gammas=None,
                  DARK_BG=True, dilate_per_scale=True,
                  verbose=True, generate_graphs=True,
                  generate_json=True, output_dir=None,
                  kernel=None, signed_frangi=False):


    raw_img = get_named_placenta(filename, maskfile=None)

    ###Multiscale & Frangi Parameters######################

    # set range of sigmas to use


    if alphas is None:
        alphas = [.15 for s in scales] # threshold constant
    if betas is None:
        betas = [0.5 for s in scales] # anisotropy constant

    # set gammas
    # declare None here to calculate half of hessian's norm
    if gammas is None:
        gammas = [None for s in scales] # structureness parameter

    ###Do preprocessing (e.g. clahe)###############
    img =  raw_img
    bg_mask = img.mask


    ###Multiscale Frangi Filter##############################

    multiscale = make_multiscale(img, scales, betas, gammas,
                                find_principal_directions=False,
                                dilate_per_scale=dilate_per_scale,
                                 kernel=kernel,
                                 signed_frangi=signed_frangi,
                                dark_bg=DARK_BG,
                                 VERBOSE=verbose)

    gammas = [scale['gamma'] for scale in multiscale]

    border_radii = [scale['border_radius'] for scale in multiscale]

    ###Process Multiscale Targets############################

    # fix targets misreported on edge of plate
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
            multiscale[i]['F'] = f.filled(0)
    else:
        for i in range(len(multiscale)):
            # harden mask (best way to do this??)
            multiscale[i]['F'] = multiscale[i]['F'].filled(0)

    ###Extract Multiscale Features############################

    pass

    ###Make Composite#########################################

    F_all = np.dstack([scale['F'] for scale in multiscale])

    if generate_graphs:

        analyze_targets(F_all, img)

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

    return F_all, img, scales, alphas


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

def analyze_targets(F_all, img):

    ###The max Frangi target##################################
    # for display purposes
    F_max = F_all.max(axis=-1)
    F_max = ma.masked_array(F_max, mask=img.mask)

    # is the frangi vesselness measure strong enough
    #F_cumulative = (F_max > alpha)


    # Variable threshold
    N = min(img.shape) // 2
    #alphas = np.logspace(-2,0, num=len(scales))*.7
    alphas = np.sqrt(1.2*scales / N)
    # try a logistic curve
    #alphas = 1 / (1+np.exp(-.2*(scales-np.sqrt(N))))
    #alphas = scales/32
    #alphas = np.logspace(-2.5,-1, num=len(scales))
    #alphas = np.linspace(0.01,1,num=len(scales))
    #alphas = np.sqrt(scales / scales.max())

    #time_of_run = datetime.datetime.now()
    #timestring = time_of_run.strftime("%y%m%d_%H%M")

    # Process Composite ###############################3

    # (deprecated, doesn't change much and takes forever)
    #matched_all = match_on_skeleton(F_cumulative, F_all)
    #wheres[np.invert(matched_all)] = 0 # first label is stuff that didn't match

    FT, wheres = apply_threshold(F_all, alpha)
    VT , wheres_VT = apply_threshold(F_all, alphas)

    ########################################
    #### THE REST IS JUST OUTPUT AND LOGGING

    print('generating outputs!')
    crop = cropped_args(img)
    """
    OUTPUT_DIR = 'output'
    base = os.path.basename(filename)

    *base, suffix = base.split('.')

    # make this its own function and just do a partial here.
    outputstub = ''.join(base) +'_' + timestring +  '_{}.'+ suffix
    outname = lambda s: os.path.join(OUTPUT_DIR, outputstub.format(s))
    """
    outname = get_outname_lambda(filename)

    # SKELETONIZED OUTPUT
    plt.imsave(outname('skel'), skeletonize(FT[crop]),
            cmap=plt.cm.gray)
    plt.imsave(outname('fmax_threshholded'), FT[crop],
            cmap=plt.cm.gray_r)
    plt.imsave(outname('fmax_variable_threshold'), VT[crop],
            cmap=plt.cm.gray_r)


    # Max Frangi score
    fig, ax = plt.subplots()
    plt.imshow(F_max[crop], cmap=plt.cm.gist_ncar)
    #plt.title(r'Max Frangi vesselness measure below threshold $\alpha={:.2f}$'.format(alpha))
    plt.title('Maximum Frangi vesselness score')
    plt.axis('off')
    c = plt.colorbar()
    c.set_ticks(np.linspace(0,1,num=11))
    plt.clim(0,1)
    plt.tight_layout()
    plt.savefig(outname('fmax'), dpi=300)

    plt.close()

    scale_label_figure(wheres, scales,
                       outname('labeled_test'),
                       crop=crop)
    scale_label_figure(wheres_VT, scales,
                       outname('labeled_VT_test'),
                       crop=crop)

    confusion_matrix = compare_trace(FT, filename=filename)

    plt.imsave(outname('confusion'), confusion_matrix[crop])

    confusion_matrix = compare_trace(VT, filename=filename)

    plt.imsave(outname('confusion_VT'), confusion_matrix[crop])



    ###Make Connected Graph##########################################

    pass

    ###Measure#######################################################

    pass

    """

"""
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
                       save_colormap_separate=False, savecolormapfile=None,
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

    # get however many samples from the colormap [R,G,B,A] array
    #tab = plt.cm.viridis_r(np.linspace(0,1,num=N))
    #tabe = np.vstack(([0,0,0,1], tab)) # add black as first entry
    #tabemap = mpl.colors.ListedColormap(tabe)

    tabemap = _build_scale_colormap(N, 'viridis_r')

    if image_only:
        plt.imsave(savefilename, wheres, cmap=tabemap, vmin=0, vmax=N)
        plt.close()
    else:
        imgplot = ax.imshow(wheres, cmap=tabemap, vmin=0, vmax=N)
        # discrete colorbar
        cbar = plt.colorbar(imgplot)

        # this is apparently hackish, beats me
        tick_locs = (np.arange(N) + 0.5)*(N-1)/N

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

    if save_colormap_separate:
        if savecolormapfile is None:
            savecolormapfile = os.path.join(output_dir, "scale_colormap.png")
        fig = plt.figure(figsize=(1,8))
        ax1 = fig.add_axes([0.05, 0.05, 0.15, 0.9])
        tick_locs = (np.arange(N) + 0.5)*(N-1)/N
        cbar = mpl.colorbar.ColorbarBase(ax1, cmap=tabemap,
                                         norm=mpl.colors.Normalize(vmin=0,
                                                                   vmax=N),
                                         orientation='vertical',
                                         ticks=tick_locs)

        plt.savefig(savecolormapfile, dpi=300)
