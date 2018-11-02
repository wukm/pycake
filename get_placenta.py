#!/usr/bin/env python3

# change this module to placenta instead of get_placenta

"""

Get registered, unpreprocessed placental images.  No automatic registration
(i.e. segmentation of placental plate) takes place here. The background,
however, *is* masked.

Again, there is no support for unregistered placental pictures.
Any region outside of the placental plate MUST be black.

There is currently no support for color images.

TODO:
    - Build sample base & organize data :v)
    - Test on many other images.
    - Think of how the interface should really work, esp for get_named_placenta
    - Fix logic in mask_background
    - Catch errors better.
    - Support for color images
    - Show a better test
    - Be able to grab trace files too.
    - Cache masked samples.
"""

import numpy as np
import numpy.ma as ma
from skimage import segmentation, morphology
import os.path
import os
import json
from scipy.ndimage import imread

def open_typefile(filename, filetype, sample_dir=None):
    """
    filetype is either 'mask' or 'trace'
    """
    # try to open what the mask *should* be named
    # this should be done less hackishly
    # for example, if filename is 'ncs.1029.jpg' then
    # this would set the maskfile as 'ncs.1029.mask.jpg'

    if filetype not in ("mask", "trace"):
        raise NotImplementedError("Can only deal with mask or trace files.")

    *base, suffix = filename.split('.')
    base = ''.join(base)
    typefile = '.'.join((base, filetype ,suffix))

    if sample_dir is None:
        sample_dir = 'samples'

    typefile = os.path.join(sample_dir, typefile)

    try:
        if filetype == 'mask':
            M = imread(typefile, mode='L')
        else:
            M = imread(typefile, mode='RGB')

    except FileNotFoundError:
        print('Could not find file', typefile)
        raise

    return M

def open_tracefile(filename, as_binary=True,
                   min_width=3, max_width=19,
                   widths=None, sample_dir=None,
                   parse_to_widths=True):
    """
    open up the trace matrix with filename 'tracefile'
    #TODO: expand this later to handle arterial traces and venous traces
    INPUT:
        tracefile:
            the name of the file

        parse_to_widths: if True, return widths. If not,
        simply return
        min_width: widths below this will be excluded (default is
                    3, the min recorded width). assuming these
                    are ints
        max_width: widths above this will be excluded (default is
                    19, the max recorded width)
        widths: an explicit list of widths that should be returned.
                in this case the above min & max are ignored.
                this way you could include widths = [3, 17, 19] only
    """

    M = open_typefile(tracefile, 'trace', sample_dir=sample_dir)

    # now a 2D matrix with binned widths 3, 5, 7, ... , 19
    T = colortrace_to_widths(M)

    if widths is None:
        min_width, max_width = int(min_width), int(max_width)
        T[T < min_width] = 0
        T[T > max_width] = 0
    else:
        # use numpy.isin(T, widths) but that's only in
        # version 1.13 and up of numpy

        # elements in A that can be found in
        # need to reshape, after v.1.13 of numpy you can use np.isin
        to_keep = np.in1d(T,x,assume_unique=True).reshape(A.shape)

        T[np.invert(to_keep)] = 0

    if as_binary:
        return T != 0
    else:
        return T



def colortrace_to_widths(T):
    """
    this will take an RGB trace image (MxNx3) and return a 2D (MxN)
    "labeled" trace corresponding to the traced pixel length.
    there is no distinguishing between arteries and vessels

    it's preferrable to do this in real-time so only one tracefile
    needs to be stored (making the sample folder less cluttered)

    Input:
        T: a MxNx3 RGB tracefile, where the colorations are assumed as
        described in NOTES below.

    Output:
        widthtrace: a MxN array whose inputs describe the width of the
        vessel (in pixels), see NOTES.

    Notes:

        The correspondence is as follows:
        3 pixels: "#ff006f",  # magenta
        5 pixels: "#a80000",  # dark red
        7 pixels: "#a800ff",  # purple
        9 pixels: "#ff00ff",  # light pink
        11 pixels: "#008aff",  # blue
        13 pixels: "#8aff00",  # green
        15 pixels: "#ffc800",  # dark yellow
        17 pixels: "#ff8a00",  # orange
        19 pixels: "#ff0015"   # bright red

    According to the original tracing protocol, the traced vessels are
    binned into these 9 sizes. Vessels with a diameter smaller than 3px
    are not traced (unless they're binned into 3px).
    """

    # a 2D picture to fix in with the pixel widths
    widthtrace = np.zeros_like(T[:,:,0])

    for pix, color in TRACE_COLORS.items():

        # get the 2D indices that are that color
        idx = np.where(np.all(T == color, axis=-1))
        widthtrace[idx] = pix

    return widthtrace

def widths_to_colors(w, show_non_matches=False):
    """
    return an RGB matrix of ints [0,255] converting back from
    [3,5,7, ... , 19] -> TRACE_COLORS

    actually making a matplotlib colormap didn't seem worth it

    this doesn't do any rounding (i.e. it ignores anything outside of
    the default widths), but maybe you'd want to?
    """
    B = np.zeros((w.shape[0], w.shape[1], 3))

    for px, rgb_triplet in TRACE_COLORS.items():
        B[w == px, : ] = rgb_triplet

    if show_non_matches:
        # everything in w not found in TRACE_COLORS will be black
        B[w == 0, : ] = (255, 255, 255)
    else:
        non_filled = (B == 0).all(axis=-1)

        B[non_filled,:] = (255,255,255) # make everything white

    # matplotlib likes the colors as [0,1], so....

    return B / 255.

TRACE_COLORS = {
    3: (255, 0, 111),
    5: (168, 0, 0),
    7: (168, 0, 255),
    9: (255, 0, 255),
    11: (0, 138, 255),
    13: (138, 255, 0),
    15: (255, 200, 0),
    17: (255, 138, 0),
    19: (255, 0, 21)}

def _hex_to_rgb(hexstring):
    """
    there's a function that does this in matplotlib.colors
    but its scaled between 0 and 1 but not even as an
    array so this is just as much work
    """
    triple = hexstring.strip("#")
    return tuple(int(x,16) for x in (triple[:2],triple[2:4],triple[4:]))

def get_named_placenta(filename, sample_dir=None, masked=True,
                       maskfile=None):
    """
    This function is to be replaced by a more ingenious/natural
    way of accessing a database of unregistered and/or registered
    placental samples.

    INPUT:
        filename: name of file (including suffix?) but NOT directory
        masked: return it masked.
        maskfile: if supplied, this use the file will use a supplied 1-channel
                mask (where 1 represents an invalid/masked pixel, and 0
                represents a valid/unmasked pixel. the supplied image must be
                the same shape as the image. if not provided, the mask is
                calculated (unless masked=False)
                the file must be located within the sample directory

                If maskfile is 'None' then this function will look for
                a default maskname with the following  pattern:

                    test.jpg -> test.mask.jpg
                    ncs.1029.jpg  -> ncs.1029.mask.jpg

        sample_directory: Relative path where sample (and mask file) is located.
                defaults to './samples'

    if masked is true (default), this returns a masked array.

    NOTE: A previous logical incongruity has been corrected. Masks should have
    1 as the invalid/background/mask value (to mask), and 0 as the
    valid/plate/foreground value (to not mask)
    """
    if sample_dir is None:
        sample_dir = 'samples'

    full_filename = os.path.join(sample_dir, filename)

    raw_img = imread(full_filename, mode='L')

    if maskfile is None:
        # try to open what the mask *should* be named
        # this should be done less hackishly
        # for example, if filename is 'ncs.1029.jpg' then
        # this would set the maskfile as 'ncs.1029.mask.jpg'
        *base, suffix = filename.split('.')
        test_maskfile = ''.join(base) + '.mask.' + suffix
        test_maskfile = os.path.join(sample_dir, test_maskfile)
        try:
            mask = imread(test_maskfile, mode='L')
        except FileNotFoundError:
            print('Could not find maskfile', test_maskfile)
            print('Please supply a maskfile. Autogeneration of mask',
                  'files is slow and buggy and therefore not supported.')
            raise
            #return mask_background(raw_img)
    else:
        # set maskfile name relative to path
        maskfile = os.path.join(sample_dir, maskfile)
        mask = imread(maskfile, mode='L')

    return ma.masked_array(raw_img, mask=mask)


def list_by_quality(quality=0, N=None, json_file=None,
                             return_empty=False):
    """
    returns a list of filenames that are of quality ``quality''

    quality is either "good" or 0
                       "OK" or 1
                       "fair" or 2
                       "bad" or 3

    N is the number of placentas to return (will return # of placentas
    of that quality or N, whichever is smaller)

    if json_name is not None just use that filename directly

    if return_empty then silenty failing is OK
    """
    if quality == 0:
        quality = 'good'
    elif quality == 1:
        quality = 'okay'
    elif quality == 2:
        quality = 'fair'
    elif quality == 3:
        quality = 'bad'
    else:
        try:
            quality = quality.lower()
        except AttributeError:
            if return_empty:
                return list()
            else:
                print(f'unknown quality {quality}')
                raise

    # json file
    if json_file is None:
        json_file = f"{quality}-mccs.json"

    # else the quality is irrelevant and hopefully the jsonfile
    # was provided
    try:
        with open(json_file, 'r') as f:
            D = json.load(f)
    except FileNotFoundError:
        if return_empty:
            return list()
        else:
            print('cannot find', json_file)
            raise

    placentas = [f'{d}.png' for d in D.keys()]

    return placentas

def check_filetype(filename, assert_png=True, assert_standard=False):
    """
    'T-BN8333878.raw.png' returns 'raw'
    'T-BN8333878.mask.png' returns 'mask'
    'T-BN8333878.png' returns 'base'

    if assert_png is True, then raise assertion error if the file
    is not of type png

    if assert_standard, then assert the filetype is
    mask, base, trace, or raw.

    etc.
    """
    basename, ext = os.path.splitext(filename)

    if ext != '.png':
        if assert_png:
            assert ext == '.png'

    sample_name, typestub = os.path.splitext(basename)

    if typestub == '':
        # it's just something like  'T-BN8333878.png'
        return 'base'
    elif typestub in ('.mask','.trace', '.raw'):
        # return 'mask' or 'trace' or 'raw'
        return typestub.strip('.')
    else:
        print('unknown filetype:', typestub)
        print('is it a weird filename?')

        print('warning: lookup failed, unknown filetype:' + typestub)

        return typestub

def list_placentas(label=None, sample_dir=None):
    """
    label is the specifier, basically just ''.startswith()

    only real use is to find all the T-BN* files

    this is hackish, if you ever decide to use a file other than
    png then this needs to change
    """

    if sample_dir is None:
        sample_dir = 'samples'

    if label is None:
        label = '' # str.startswith('') is always True

    placentas = list()

    for f in os.listdir(sample_dir):

        if f.startswith(label):
            # oh man they gotta be png files
            if check_filetype(f) == 'base':
                placentas.append(f)

    return sorted(placentas)

def mask_background(img):
    """
    Warning: this function is slow and buggy and therefore deprecated
    as "out of scope". Please fix or remove.

    Masks all regions of the image outside the placental plate.

    INPUT:
        img:
            A color or grayscale array corresponding to an image of a placenta
            with the plate in the 'middle.' Outer regions should be black.

    OUTPUT:
        masked_img:
            A numpy.ma.masked_array with the same dimensions.
    """
    print("""
          Warning, this function is slow and buggy and therefore
          deprecated. Please supply a mask file yourself.
          """
          )

    if img.ndim == 3:

        #mark any pixel with with content in any channel
        bg_mask = img.any(axis=-1)
        bg_mask = np.invert(bg_mask)

        # make the mask multichannel to match dim of input
        bg_mask = np.repeat(bg_mask[:,:,np.newaxis], 3, axis=2)

    else:

        # same as above
        bg_mask = (img != 0)
        bg_mask = np.invert(bg_mask)

    # the above approach will probably work for any real image (i.e. a
    # photgraph). it will obviously fail for any image where there is true black
    # in the placental plane. This should work instead:

    # find the outer boundary and mark outside of it.
    # run with defaults, sufficient
    bound = morphology.convex_hull_image(bg_mask)
    bound = segmentation.find_boundaries(bg_mask, mode='inner', background=1)
    bg_mask[bound] = 1

    #remove any small holes found inside the plate (regions or single pixels
    #that happen to be black).  run with defaults, sufficient
    holes = morphology.remove_small_holes(bg_mask)
    bg_mask[holes] = 1

    return ma.masked_array(img, mask=bg_mask)

def show_mask(img):
    """
    show a masked grayscale image with a dark blue masked region

    custom version of imshow that shows grayscale images with the right colormap
    and, if they're masked arrays, sets makes the mask a dark blue)
    a better function might make the grayscale value dark blue
    (so there's no confusion)
    """

    from numpy.ma import is_masked
    from skimage.color import gray2rgb
    import matplotlib.pyplot as plt


    if not is_masked(img):
        plt.imshow(img, cmap=plt.cm.gray)
    else:

        mimg = gray2rgb(img.filled(0))
        # fill blue channel with a relatively dark value for masked elements
        mimg[img.mask, 2] = 60
        plt.imshow(mimg)

if __name__ == "__main__":

    """test that this works on an easy image."""

    from scipy.ndimage import imread
    import matplotlib.pyplot as plt
    test_filename = 'barium1.png'
    #test_maskfile = 'barium1.mask.png'

    img =  get_named_placenta(test_filename, maskfile=None)

    print('showing the mask of', test_filename)
    print('run plt.show() to see masked output')
    show_mask(img)


def _cropped_bounds(img, mask=None):

    if mask is not None:

        img = ma.masked_array(img, mask=mask)

    X,Y = (np.argwhere(np.invert(img.mask).any(axis=k)).squeeze() for k in (0,1))

    if X.size == 0:
        X = [None,None] # these will slice correctly
    if Y.size == 0:
        Y = [None,None]

    return Y[0],Y[-1],X[0],X[-1]

def cropped_args(img, mask=None):
    """
    get a slice that would crop image
    i.e. img[cropped_args(img)] would be a cropped view
    """

    x0, x1, y0,y1 = _cropped_bounds(img, mask=None)

    return np.s_[x0:x1,y0:y1]

def cropped_view(img, mask=None):
    """
    removes entire masked rows and columns from the borders of a masked array.
    will return a masked array of smaller size

    don't ask me about data

    the name sucks too
    """

    # find first and last row with content
    x0, x1, y0,y1 = _cropped_bounds(img, mask=mask)

    return img[x0:x1,y0:y1]
