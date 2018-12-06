#!/usr/bin/env python3

from placenta import (get_named_placenta, list_placentas, open_typefile,
                      show_mask)
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import mask_stump
import numpy.ma as ma
import sys
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_int

for n, filename in enumerate(list_placentas('T-BN')):

    print(filename)

    raw = open_typefile(filename, 'raw')
    #raw = (equalize_adapthist(raw))
    img = get_named_placenta(filename)
    
    stump = mask_stump(raw, mask=img.mask, mask_only=True)
    
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(30,12))

    ax[0].imshow(raw)
    ax[0].set_title(filename)

    newmask = show_mask(ma.masked_array(img, mask=stump))
    ax[1].imshow(newmask)
    ax[2].imshow(stump*1. + img.mask*2, vmin=0, vmax=3)

    plt.show()
    
    if n>0 and not n%10:
        if input('make more?') == 'n':
            sys.exit(0)
