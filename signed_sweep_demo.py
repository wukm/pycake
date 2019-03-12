#!/usr/bin/env python3

"""
this is analogous to scale_sweep_demo.py. should do the same thing but
for signed Frangi
"""

from placenta import (get_named_placenta, list_placentas, _cropped_bounds,
                      cropped_view, cropped_args, show_mask)
from frangi import frangi_from_image

import numpy as np
import numpy.ma as ma

from plate_morphology import dilate_boundary

import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product

# pick two samples and two insets (should be the same size)
demo1 = 'BN2315363', np.s_[370:670, 530:930]
demo2 = 'BN5280796', np.s_[150:450, 530:930]

for sample_name, inset_slice in (demo1, demo2):
    img = get_named_placenta(f'T-{sample_name}.png')

    crop = cropped_args(img)

    F, fi = list(), list() # make some empty lists to store for inspection

    #scales = np.logspace(-3, 4, num=8, base=2)
    scales = [0.2, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
    CMAP = plt.cm.Spectral
    cmin, cmax = (-0.4, 0.4)

    for n, sigma in enumerate(scales):
        R = max(int(sigma*3), 10)
        target = frangi_from_image(img, sigma, dark_bg=False,
                                   signed_frangi=True, dilation_radius=R)
        plate = target[crop].filled(0)
        inset = target[inset_slice].filled(0)
        F.append(plate)
        fi.append(inset)
        #for label in ['plate', 'inset']:
        #    if label == 'inset':
        #        printable = inset
        #    else:
        #        printable = plate

        #    plt.imshow(printable, cmap=CMAP)
        #    plt.title(r'$\sigma={:.2f}$'.format(sigma))
        #    plt.tight_layout()
        #    c = plt.colorbar()
        #    c.set_ticks = np.linspace(cmin, cmax, num=len(scales)+1)
        #    plt.clim(cmin, cmax)
        #    plt.axis('off')
        #    outname = f'demo_output/signsweep_{sample_name}_{label}_{n}.png'
        #    #plt.savefig(outname, dpi=300, bbox_inches='tight')
        #    print('saved', outname)
        #    plt.close()

    # now make a stitched together version
    for label in ['plate', 'inset']:
        if label == 'inset':
            L = fi
            imgview = img[inset_slice].filled(0)
            figsize = (12, 6)

        else:
            L = F
            imgview = img[crop].filled(0)
            figsize = (12, 9)
        #adjust this manually depending on how many scales you end up using!

        nrows, ncols = 2, 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for n, (i, j) in enumerate(product(range(nrows), range(ncols))):

            if n == 0:
                axes[i,j].imshow(imgview, cmap=plt.cm.gray)
                axes[i,j].set_title('raw')
            else:
                im = axes[i,j].imshow(L[n], cmap=CMAP, vmin=cmin, vmax=cmax)
                axes[i,j].set_title(r'$\sigma={:.2f}$'.format(scales[n]))

            plt.setp(axes[i,j].get_xticklabels(), visible=False)
            plt.setp(axes[i,j].get_yticklabels(), visible=False)

        #for i in range(5):
        #    fig.tight_layout()

        #fig.subplots_adjust(right=0.8)
        #cax = fig.add_axes([.85,.15,.05,.7])
        #c = fig.colorbar(im, ax=cax)

        fig.subplots_adjust(top=0.954, bottom=0.025, left=0.010,
                            right=0.989, hspace=0.0, wspace=0.0)

        plt.savefig(f'demo_output/signsweep_stitch_{sample_name}_{label}.png',
                    dpi=300)

        cfile = 'demo_output/signsweep_colorbar.png'
        if os.path.isfile(cfile):
            continue # no need to make another

        fig = plt.figure(figsize=(figsize[0],2))
        ax1 = fig.add_axes([0.15, 0.25, 0.75, 0.5])
        cbar = mpl.colorbar.ColorbarBase(ax1, cmap=CMAP,
                                        norm=mpl.colors.Normalize(cmin,cmax),
                                        orientation='horizontal')
        plt.savefig(cfile, dpi=300)
        #top = np.concatenate(L[:4],axis=1)
        #bottom = np.concatenate(L[4:],axis=1)
        #stitched = np.concatenate((top,bottom),axis=0)
        #imga = plt.imshow(stitched, cmap=CMAP)
        #plt.imsave(f'demo_output/signsweep_stitch_{sample_name}_{label}.png',
        #        stitched, cmap=CMAP, vmin=cmin, vmax=cmax)

        # also save the original pic
        #plt.imsave(f'demo_output/signsweep_{sample_name}_{label}_raw',
        #        imgview, cmap=plt.cm.gray)
