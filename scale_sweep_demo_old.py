#!/usr/bin/env python3

from placenta import (get_named_placenta, list_placentas, _cropped_bounds,
                      cropped_view, cropped_args, show_mask)
from frangi import frangi_from_image

import numpy as np
import numpy.ma as ma

from plate_morphology import dilate_boundary

import os.path
import matplotlib.pyplot as plt

# pick two samples and two insets (should be the same size)
demo1 = 'BN2315363', np.s_[370:670, 530:930]
demo2 = 'BN5280796', np.s_[150:450, 530:930]

for sample_name, inset_slice in (demo1, demo2):
    img = get_named_placenta(f'T-{sample_name}.png')


    F, fi = list(), list() # make some empty lists to store for inspection

    # struggling with this now. sigma=1.23
    #scales = np.logspace(-3, 4, num=8, base=2)
    scales = [0.2, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
    #CMAP = plt.cm.nipy_spectral_r
    #CMAP = plt.cm.nipy_spectral
    CMAP = plt.cm.viridis
    cmin, cmax = (0, 0.4)

    for n, sigma in enumerate(scales):
        target = frangi_from_image(img, sigma, dark_bg=False,
                                   dilation_radius=10)
        plate = cropped_view(target).filled(0)
        inset = target[inset_slice]
        F.append(plate)
        fi.append(inset)
        for label in ['plate', 'inset']:
            if label == 'inset':
                printable = inset
            else:
                printable = plate

            plt.imshow(printable, cmap=CMAP)
            plt.title(r'$\sigma={:.2f}$'.format(sigma))
            plt.tight_layout()
            c = plt.colorbar()
            c.set_ticks = np.linspace(cmin, cmax, num=len(scales)+1)
            plt.clim(cmin, cmax)
            plt.axis('off')
            outname = f'demo_output/scalesweep_{sample_name}_{label}_{n}.png'
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            print('saved', outname)
            plt.close()


    # now make a stitched together version
    for label in ['plate', 'inset']:
        if label == 'inset':
            L = fi
            imgview = img[inset_slice].filled(0)
        else:
            L = F
            imgview = cropped_view(img).filled(0)
        #adjust this manually depending on how many scales you end up using!
        top = np.concatenate(L[:4],axis=1)
        bottom = np.concatenate(L[4:],axis=1)
        stitched = np.concatenate((top,bottom),axis=0)
        imga = plt.imshow(stitched, cmap=CMAP)
        plt.imsave(f'demo_output/scalesweep_stitch_{sample_name}_{label}.png',
                stitched, cmap=CMAP, vmin=cmin, vmax=cmax)

        # also save the original pic
        plt.imsave(f'demo_output/scalesweep_{sample_name}_{label}_raw',
                imgview, cmap=plt.cm.gray)
