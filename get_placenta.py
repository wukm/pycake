#!/usr/bin/env python3

##TODO change the name of this module to something more descriptive
"""
Get registered, unpreprocessed placental images.  No automatic registration
(i.e. segmentation of placental plate) takes place here. The background,
however, *is* masked.

Again, there is no support for unregistered placental pictures.
A mask file must be provided.

There is currently no support for color images.
"""

import numpy as np
import numpy.ma as ma
from skimage import segmentation, morphology
import os.path
import os
import json
from scipy.ndimage import imread

def open_typefile(filename, filetype, sample_dir=None, mode=None):
    """
    filetype is either 'mask' or 'trace'
    mask -> 'L' mode
    trace -> 'RGB' mode
    use mode keyword to override this behavior (for example if you
    want a binary trace)

    typefiles that aren't the above will be treated as 'L'
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

    if mode is not None:
        if filetype == 'mask':
            mode = 'L'
        elif filetype == 'trace':
            mode = 'RGB'
        else:
            # handle this if you need to?
            mode = 'L'
    try:
        img = imread(typefile, mode=mode)

    except FileNotFoundError:
        print('Could not find file', typefile)
        raise

    return img


def open_tracefile(base_filename, as_binary=True,
                   sample_dir=None):

    """

    ###width parsing is no longer done here. instead, this function
    should handle the venous/arterial difference.

    this currently only serves to open the RGB traces as binary
    files instead of RGB, which is processed later

    #TODO: expand this later to handle arterial traces and venous traces
    INPUT:
        base_filename: the name of the base file, not the tracefile itself
        as_binary: if True
    """

    if as_binary:
        mode = 'L'
    else:
        mode = 'RGB'

    T = open_typefile(filename, filetype, sample_dir=None, mode=mode)

    if as_binary:

        return T != 0

    else:

        return T





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
