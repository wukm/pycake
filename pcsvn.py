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
import os.path
import json
import datetime

def make_multiscale(img, scales, betas, gammas, find_principal_directions=False,
                    dilate=True, dark_bg=True, VERBOSE=True):
    """returns an ordered list of dictionaries for each scale
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
        if dilate:
            if sigma < 2.5:
                radius = 10
            else:
                radius = int(sigma*4) # a little aggressive
        else:
            radius = None

        if VERBOSE:
            print('σ={}'.format(sigma))
            print('finding hessian')

        # get hessian components at each pixel as a triplet (Lxx, Lxy, Lyy)
        print("file type is:", img.dtype)
        hesh = fft_hessian(img, sigma)

        if VERBOSE:
            print('finding principal curvatures')


        k1, k2 = principal_curvatures(img, sigma=sigma, H=hesh)

        # area of influence to zero out
        if dilate:
            collar = dilate_boundary(None, radius=radius, mask=img.mask)

            k1[collar] = 0
            k2[collar] = 0

        # set anisotropy parameter if not specified
        if gamma is None:
            # Frangi suggested 'half the max Hessian norm' as an empirical
            # half the max spectral radius is easier to calculate so do that
            # shouldn't be affected by mask data but should make sure the
            # mask is *well* far away from perimeter
            gamma_alt = .5 * np.abs(k2).max()
            print("half of k2 max is", gamma_alt)

            # or actually calculate half of max hessian norm
            # using frob norm = sqrt(trace(AA^T))
            hxx, hxy, hyy = hesh
            max_hessian_norm = np.sqrt((hxx**2 + 2*hxy**2 + hyy**2).max())
            gamma = .5*max_hessian_norm
            print("half of max hessian norm is", gamma)
        if VERBOSE:
            print('finding Frangi targets with β={} and γ={:.2}'.format(beta, gamma))

        # calculate frangi targets at this scale
        targets = get_frangi_targets(k1,k2,
                    beta=beta, gamma=gamma, dark_bg=dark_bg, threshold=False)

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

def match_on_skeleton(skeleton_of, layers, VERBOSE=True):
    """using the computed skeleton of ``skeleton_of'',
    return a composite image where blobs layers are incrementally added to the
    composite image if that blob coincides at some location of the skeleton
    """

    if ma.is_masked(skeleton_of):
        skeleton_of = skeleton_of.filled(0)

    skel = skeletonize(skeleton_of)
    matched_all = np.zeros_like(skel)

    # in reverse order (largest to smallest)
    for n in range(layers.shape[-1]-1, -1, -1):
        print('matching in layer #{}'.format(n))
        current_layer = layers[:,:,n]

        # only care about things in the current layer above the mean of that
        # layer (AAAH)
        current_layer = current_layer > current_layer.mean()
        #current_layer = current_layer > 0.2

        # don't match anything that's been matched already
        current_layer[matched_all] = 0

        # label each connected blob
        el, nl = label(current_layer, return_num=True)
        matched = np.zeros_like(current_layer)

        for region in range(1, nl+1):
            if np.logical_and(el==region, skel).any():
                matched = np.logical_or(matched, el==region)

        matched_all = np.logical_or(matched_all, matched)

    return matched_all

def extract_pcsvn(filename, alpha=.15, log_range=(0,4.5), DARK_BG=True,
                   dilate_per_scale=True, n_scales = 20, verbose=True):

    raw_img = get_named_placenta(filename, maskfile=None)

    ###Multiscale & Frangi Parameters######################

    # set range of sigmas to use

    log_min, log_max = log_range
    #log_min = -1 # minimum scale is 2**log_min
    #log_max = 4.5 # maximum scale is 2**log_max
    n_scales = 20
    scales = np.logspace(log_min, log_max, n_scales, base=2)


    #alpha = 0.15 # Threshold for vesselness measure
    betas = [0.5 for s in scales] #anisotropy measure

    # set gammas
    # declare None here to calculate half of hessian's norm
    gammas = [None for s in scales] # structureness parameter

    ###Do preprocessing (e.g. clahe)###############
    img =  raw_img
    bg_mask = img.mask

    ###Logging#####################################
    if verbose:
        print(" Running pcsvn.py on the image file", filename,
                "with frangi parameters as follows:")
        print("alpha (vesselness threshold): ", alpha)
        print("scales:", scales)
        print("betas:", betas)
        print("gammas will be calculated as half of hessian norm")

    ###Multiscale Frangi Filter##############################

    multiscale = make_multiscale(img, scales, betas, gammas,
                                find_principal_directions=False,
                                dilate=dilate_per_scale,
                                dark_bg=DARK_BG)

    gammas = [scale['gamma'] for scale in multiscale]

    border_radii = [scale['border_radius'] for scale in multiscale]
    ###Process Multiscale Targets############################

    # fix targets misreported on edge of plate
    # wait are we doing this twice?
    if dilate_per_scale:
        print('trimming collars of plates (per scale)')

        for i in range(len(multiscale)):
            f = multiscale[i]['F']
            # twice the buffer (be conservative!)
            radius = int(multiscale[i]['sigma']*2)
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

    ###The max Frangi target##################################

    # scale???
    #F_all = F_all* (scales**-1)

    F_max = F_all.max(axis=-1)

    F_max = ma.masked_array(F_max, mask=img.mask)

    # is the frangi vesselness measure strong enough
    F_cumulative = (F_max > alpha)

    # ALPHA RANGE?
    N = min(img.shape) // 2
    #alphas = np.logspace(-2,0, num=len(scales))*.7
    alphas = np.sqrt(1.2*scales / N)
    # try a logistic curve
    #alphas = 1 / (1+np.exp(-.2*(scales-np.sqrt(N))))
    #alphas = scales/32
    print(alphas)
    #alphas = np.logspace(-2.5,-1, num=len(scales))
    #alphas = np.linspace(0.01,1,num=len(scales))

    #alphas = np.sqrt(scales / scales.max())

    time_of_run = datetime.datetime.now()
    timestring = time_of_run.strftime("%y%m%d_%H%M")

    # Process Composite ###############################3

    # (deprecated, doesn't change much and takes forever)
    #matched_all = match_on_skeleton(F_cumulative, F_all)
    #wheres[np.invert(matched_all)] = 0 # first label is stuff that didn't match


    # assign a label for where each max was found
    wheres = F_all.argmax(axis=-1)
    wheres += 1 # zero where no match
    wheres[F_max < alpha] = 0

    # using variable alpha model?

    F_VT = F_all.copy()
    F_VT[F_all < alphas] = 0

    wheres_VT = F_VT.argmax(axis=-1)
    wheres_VT += 1
    # zero all pixels that never passed the threshold a t any level
    wheres_VT[(F_all < alphas).all(axis=-1)] = 0
    VT = (wheres_VT != 0) # this is the result
    # pixels where any of the thresholds have been passed
    #VT = (F_all > alphas).any(axis=-1)

    #wheres_VT[np.invert(VT)] = 0

    # use a colorscheme where you can see each later clearly
    #plt.imshow(wheres, cmap=plt.cm.tab20b)
    #plt.colorbar()
    #plt.show()
    OUTPUT_DIR = 'output'
    base = os.path.basename(filename)

    *base, suffix = base.split('.')

    # make this its own function and just do a partial here.
    outname = lambda s: os.path.join(OUTPUT_DIR,
                                ''.join(base) +'_' + timestring +  '_' + s + '.'+ suffix)

    plt.imsave(outname('skel'), skeletonize(F_cumulative.filled(0)),
            cmap=plt.cm.gray)
    plt.imsave(outname('fmax_threshholded'), F_cumulative.filled(0),
            cmap=plt.cm.gray_r)
    plt.imsave(outname('fmax_variable_threshold'), wheres_VT > 0,
            cmap=plt.cm.gray_r)


    # Max Frangi score
    fig, ax = plt.subplots(figsize=(12,8))
    plt.imshow(F_max, cmap=plt.cm.gist_ncar)
    #plt.title(r'Max Frangi vesselness measure below threshold $\alpha={:.2f}$'.format(alpha))
    plt.title('Maximum Frangi vesselness score')
    plt.axis('off')
    c = plt.colorbar()
    c.set_ticks(np.linspace(0,1,num=11))
    plt.clim(0,1)
    plt.tight_layout()
    plt.savefig(outname('fmax'), dpi=300)

    plt.close()

    ### MAKE SCALE LABEL GRAPH
    # discrete colorbar adapted from https://stackoverflow.com/a/50314773
    fig, ax = plt.subplots(figsize=(12,8)) # not sure about figsize
    N = len(scales)+1 # number of scales / labels
    cmap = plt.get_cmap('nipy_spectral', N) # discrete sample of color map

    imgplot = ax.imshow(wheres, cmap=cmap)

    # discrete colorbar
    cbar = plt.colorbar(imgplot)

    # this is apparently hackish, beats me
    tick_locs = (np.arange(N) + 0.5)*(N-1)/N

    cbar.set_ticks(tick_locs)
    # label each tick with the sigma value
    scalelabels = [r"$\sigma = {:.2f}$".format(s) for s in scales]
    scalelabels.insert(0, "(no match)")
    # label with their label number (or change this to actual sigma value
    cbar.set_ticklabels(scalelabels)
    ax.set_title(r"Scale ($\sigma$) of maximum vesselness ")

    plt.savefig(outname('labeled'), dpi=300)
    plt.tight_layout()
    plt.close()

    # discrete colorbar adapted from https://stackoverflow.com/a/50314773
    fig, ax = plt.subplots(figsize=(12,8)) # not sure about figsize
    N = len(scales)+1 # number of scales / labels
    cmap = plt.get_cmap('nipy_spectral', N) # discrete sample of color map

    imgplot = ax.imshow(wheres_VT, cmap=cmap)

    # discrete colorbar
    cbar = plt.colorbar(imgplot)

    # this is apparently hackish, beats me
    tick_locs = (np.arange(N) + 0.5)*(N-1)/N

    cbar.set_ticks(tick_locs)
    # label each tick with the sigma value
    scalelabels = [r"$\sigma = {:.2f}$".format(s) for s in scales]
    scalelabels.insert(0, "(no match)")
    # label with their label number (or change this to actual sigma value
    cbar.set_ticklabels(scalelabels)
    ax.set_title(r"Scale ($\sigma$) of maximum vesselness ")

    plt.savefig(outname('labeled_VT'), dpi=300)
    plt.tight_layout()
    plt.close()

    confusion_matrix = compare_trace(F_cumulative.filled(0), filename=filename)

    plt.imsave(outname('confusion'), confusion_matrix)

    confusion_matrix = compare_trace(VT, filename=filename)

    plt.imsave(outname('confusion_VT'), confusion_matrix)

    logdata = {'time': timestring,
            'filename': filename,
            'DARK_BG': DARK_BG,
            'fixed_alpha': alpha,
            'VT_alphas': list(alphas),
            'betas': betas,
            'gammas': gammas,
            'sigmas': list(scales),
            'log_min': log_min,
            'log_max': log_max,
            'n_scales': n_scales,
            'border_radii': border_radii
            }

    dumpfile = os.path.join(OUTPUT_DIR,
                            ''.join(base) + '_' + str(timestring)
                            + '.json')
    with open(dumpfile, 'w') as f:
        json.dump(logdata, f, indent=True)

    # list of each scale's frangi targets for easier introspection
    Fs = [F_all[:,:,j] for j in range(F_all.shape[-1])]

    ###Make Connected Graph##########################################

    pass

    ###Measure#######################################################

    pass

    return Fs, locals()


if __name__ == "__main__":


    from get_placenta import show_mask as mimshow
    from get_placenta import list_placentas
    show = plt.show
    imshow = plt.imshow


    #################################################################
    #### MAIN LOOP ##################################################
    #################################################################
    #################################################################

    ###Static Parameters############################

    #VERBOSE=False

    ###Set base image#############################

    placentas = list_placentas('T-BN')[:15]
    N_samples = len(placentas)
    print(N_samples, "samples total!")
    for i, filename in enumerate(placentas):
        print('*'*80)
        print('extracting PCSVN of', filename,
              '\t ({} of {})'.format(i,N_samples))

        alpha = .08
        DARK_BG = False
        log_range = (-2,3)
        dilate_per_scale = False
        extract_pcsvn(filename, DARK_BG=DARK_BG,
                          alpha=alpha, log_range=log_range,
                          dilate_per_scale = False)

