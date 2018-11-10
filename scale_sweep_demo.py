#!/usr/bin/env python3

from placenta import get_named_placenta, list_placentas, _cropped_bounds, cropped_view, cropped_args, show_mask
from frangi import frangi_from_image

import numpy as np
import numpy.ma as ma
from plate_morphology import dilate_boundary

import os.path
import matplotlib.pyplot as plt

#imgfile = list_placentas('T-BN')[32]

img = get_named_placenta('T-BN2315363.png')

img = dilate_boundary(img,radius=5)
F = list()
fi = list()

#scales = np.logspace(-3,3,base=2,num=8)
#scales = np.linspace(.25,8,num=8)

scales = np.linspace(.25,4,num=6)
for n, sigma in enumerate(scales, 1):
    target = frangi_from_image(img, sigma, dark_bg=False)
    plate = cropped_view(target).filled(0)
    inset = target[370:660,530:900]
    F.append(plate)
    fi.append(inset)
    for label in ['plate','inset']:
        if label == 'inset':
            printable = inset
        else:
            printable = plate

        plt.imshow(printable, cmap=plt.cm.gist_earth)
        plt.title(r'$\sigma={:.2f}$'.format(sigma))
        plt.tight_layout()
        c = plt.colorbar()
        c.set_ticks = np.linspace(0,0.6, num=7)
        plt.clim(0,0.6)
        outname = 'demo_output/scalesweep_{}_{}.png'.format(n,label)
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        print('saved', outname)
        plt.close()

# now make a stitched together version
for label in ['plate', 'inset']:
    if label == 'inset':
        L = fi
    else:
        L = F
    top = np.concatenate(L[:3],axis=1)
    bottom = np.concatenate(L[3:],axis=1)
    stitched = np.concatenate((top,bottom),axis=0)
    imga = plt.imshow(stitched, cmap=plt.cm.gist_earth)
    plt.imsave('demo_output/sweep_stitched_{}.png'.format(label),
               stitched, cmap=plt.cm.gist_earth)
    #plt.colorbar(); plt.clim(0,0.3)


