#!/usr/bin/env python3

from get_placenta import get_named_placenta
from hfft import fft_hessian
from diffgeo import principal_curvatures, principal_directions
from frangi import get_frangi_targets
from vessel_filters import rotating_box_filter
import numpy as np
import numpy.ma as ma
import pickle

from skimage.morphology import label, skeletonize, disk, binary_erosion, convex_hull_image, binary_dilation
from skimage.segmentation import find_boundaries


def make_multiscale(img, scales, betas, gammas, find_principal_directions=False,
                    dark_bg=True, VERBOSE=True, trim_first=False):
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

    for i, sigma, beta, gamma in zip(range(len(scales)), scales, betas, gammas):

        if sigma < 5:
            radius = 5
        else:
            radius = int(sigma*2.5) # a little conservative
        collar = dilate_plate(img, radius=radius, plate_mask=img.mask)

        collar = collar.mask
        if VERBOSE:
            print('σ={}'.format(sigma))
            print('finding hessian')

            # get hessian components at each pixel as a triplet (Lxx, Lxy, Lyy)
        hesh = fft_hessian(img, sigma)

        if VERBOSE:
            print('finding principal curvatures')

        k1, k2 = principal_curvatures(img, sigma=sigma, H=hesh)

        k1[collar] = 0
        k2[collar] = 0

        # set anisotropy parameter if not specified
        if gamma is None:
            # Frangi suggested 'half the max Hessian norm' as an empirical
            # half the max spectral radius is easier to calculate so do that

            # shouldn't be affected by mask data but should make sure the
            # mask is *well* far away from perimeter
            gamma = .5 * np.abs(k2).max()

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
                      'k2': k2}

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

def erode_plate(img, erosion_radius=20, plate_mask=None):
    """Manually remove (erode) a region around the plate
    This doesn't work well and runs way too long for the simplicity
    of the task (brew your own erode convex image instead).
    You need to self-supply radius."""

    if plate_mask is None:
        # grab the mask from input image
        try:
            plate_mask = img.mask
        except AttributeError:
            raise('Need to supply mask information')

    # convex_hull_image finds white pixels
    plate_mask = np.invert(plate_mask)

    # find convex hull of mask (make erosion calculation easier)
    plate_mask = np.invert(convex_hull_image(plate_mask))

    # this is much faster than a disk. a plus sign might be better even.
    #selem = square(erosion_radius)
    # also this correctly has erosion radius as the RADIUS!
    # input for square() and disk() is the diameter!
    selem = np.zeros((erosion_radius*2 + 1, erosion_radius*2 + 1),
                        dtype='bool')
    selem[erosion_radius,:] = 1
    selem[:,erosion_radius] = 1
    eroded_mask = binary_erosion(plate_mask, selem=selem)

    # this is by default additive with whatever
    return ma.masked_array(img, mask=eroded_mask)

def dilate_plate(img, radius=10, plate_mask=None):
    """
    grows the mask by a specified radius
    can this return a masked image rather than return a new one
    to be more memory efficient? or should this just return a
    mask to be applied?
    """

    if plate_mask is None:
        # grab the mask from input image
        try:
            plate_mask = img.mask
        except AttributeError:
            raise('Need to supply mask information')

    perimeter = find_boundaries(plate_mask)

    # this is extremely inefficient. can it just convolve around
    # the loop and skip convolving areas where everything is 0?
    # maskpad = binary_dilation(perimeter, selem=disk(radius))

    # THIS IS WAY BETTER!
    maskpad = np.zeros_like(perimeter)

    M,N = maskpad.shape
    for i,j in np.argwhere(perimeter):
        # just make a cross shape on each of those points
        # these will silently fail if slice is OOB
        maskpad[max(i-radius,0):min(i+radius,M),j] = 1
        maskpad[i,max(j-radius,0):min(j+radius,N)] = 1

    new_mask = np.bitwise_or(maskpad, plate_mask)
    # this is by default additive with whatever mask img already has
    # unless it's not...?
    return ma.masked_array(img, mask=new_mask)
#################################################################
#### MAIN LOOP ##################################################
#################################################################
#################################################################

###Static Parameters############################

# main vesselness threshold
alpha = 0.08

###Set base image#############################

#filename = 'barium1.png'; DARK_BG = True
#filename = 'barium2.png'; DARK_BG = True
#filename = 'NYMH_ID130016i.png'; DARK_BG = True
#filename = 'NYMH_ID130016u.png'; DARK_BG = False
#filename = 'NYMH_ID130016u_inset.png'; DARK_BG = False
#filename =  'im0059.png'; DARK_BG = False # set alpha much smaller, like .1

raw_img = get_named_placenta(filename, maskfile=None)

###Do preprocessing (e.g. clahe)###############

EARLY_TRIM=False

print('Note: no preprocessing is done anymore.')
img =  raw_img
bg_mask = img.mask

###Set Parameter(s) for Frangi#################

# set range of sigmas to use (declare these above)
# the
log_min = 1 # minimum scale is 2**log_min
log_max = 4.5 # maximum scale is 2**log_max
scales = np.logspace(log_min, log_max, num=10, base=2)


# set betas (anisotropy parameters)
# 0.5 is frangi's recommendation... i think
betas = [0.6 for s in scales]

# set gammas (structness parameter)
# declare None here to calculate half of hessian's norm
gammas = [None for s in scales]


###Logging#####################################

print(" Running pcsvn.py on the image file", filename,
        "with frangi parameters as follows:")
print("alpha (vesselness threshold): ", alpha)
print("scales:", scales)
print("betas:", betas)
print("gammas will be calculated as half of hessian norm")

# Multiscale Frangi Filter

multiscale = make_multiscale(img, scales, betas, gammas,
                             find_principal_directions=False,
                             dark_bg=DARK_BG,
                             trim_first=EARLY_TRIM)

#with open('barium2_multiscale_171024.pkl', 'rb') as f:
    #multiscale = pickle.load(f)

# Process Multiscale Targets

# fix targets misreported on edge of plate

if not EARLY_TRIM:
    print('trimming collars of plates (per scale)')

    for i in range(len(multiscale)):
        f = multiscale[i]['F']
        # twice the buffer (be conservative!)
        radius = int(multiscale[i]['sigma']*2)
        print('dilating plate for radius={}'.format(radius))
        f = dilate_plate(f, radius=radius, plate_mask=img.mask)
        multiscale[i]['F'] = f.filled(0)


# Extract Multiscale Features

pass

# Make Composite

F_all = np.dstack([scale['F'] for scale in multiscale])

# the max Frangi target
F_max = F_all.max(axis=-1)

F_max = ma.masked_array(F_max, mask=img.mask)

# is the frangi vesselness measure strong enough
F_cumulative = (F_max > alpha)

# where there's any match at all (LOOK INTO THIS)
#matched_all = match_on_skeleton(F_cumulative, F_all)

# Process Composite ###############################3
# make this a function

# assign an index (later color) for where each max was found
#wheres = F_all.argmax(axis=-1)

#wheres += 1

#wheres[np.invert(matched_all)] = 0 # first label is stuff that didn't match
#wheres[F_max < alpha] = 0 # or that didn't pass the threshold

# same thing w/o skeleton matching
wheres = F_all.argmax(axis=-1)
wheres += 1
wheres[F_max < alpha] = 0

# Make Connected Graph

pass

# Measure

pass

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os.path

    from get_placenta import show_mask as mimshow
    show = plt.show
    imshow = plt.imshow

    # use a colorscheme where you can see each later clearly
    #plt.imshow(wheres, cmap=plt.cm.tab20b)
    #plt.colorbar()
    #plt.show()
    OUTPUT_DIR = 'output'
    base = os.path.basename(filename)

    *base, suffix = base.split('.')

    # make this its own function and just do a partial here.
    outname = lambda s: os.path.join(OUTPUT_DIR,
                                ''.join(base) + '_' + s + '.'+ suffix)
    #plt.imsave(base +'_scales_whole.png', wheres, cmap=plt.cm.tab20b)
    #plt.imsave(base +'_scales_whole_noskel.png', wheres, cmap=plt.cm.tab20b)
    #plt.imsave(base +'_fmax.png', F_max.filled(0), cmap=plt.cm.Blues)
    #plt.imsave(base +'_skel.png', skeletonize(F_cumulative.filled(0)),
    #           cmap=plt.cm.gray)
    plt.imsave(outname('skel'), skeletonize(F_cumulative.filled(0)),
               cmap=plt.cm.gray)
    plt.imsave(outname('fmax_threshholded'), F_cumulative.filled(0))

    plt.imshow(F_max.filled(0), cmap=plt.cm.gist_ncar)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outname('fmax'), dpi=300)

    plt.close()

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
    ax.set_title(r"scale ($\sigma$) of matched targets")

    plt.savefig(outname('labeled'), dpi=300)
    # list of each scale's frangi targets for easier introspection
    Fs = [F_all[:,:,j] for j in range(F_all.shape[-1])]
