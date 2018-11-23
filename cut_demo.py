#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from placenta import (open_typefile, list_placentas, get_named_placenta)
from plate_morphology import mask_cuts, dilate_boundary

from skimage.color import gray2rgb
from skimage.morphology import thin, binary_dilation, disk
import numpy.ma as ma


def l2_dist(p,q):
    return int(np.round(np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)))

placentas = list_placentas('T-BN')
samples_with_cuts = list()

for filename in list_placentas('T-BN'):
    
    #if filename != "T-BN2459820.png":
    #       continue

    img = get_named_placenta(filename)
    ucip = open_typefile(filename, 'ucip')
    
    C, has_cut = mask_cuts(img, ucip, return_success=True, in_place=False)
    
    if has_cut:
        
        dilcut = img.copy()

        print(filename, "has a cut!")
        samples_with_cuts.append(filename)
        
        B = np.all(ucip==(0,0,255), axis=-1)
        G = np.all(ucip==(0,255,0), axis=-1)
        
        cutmarks = np.nonzero(thin(B))
        perimeter = np.nonzero(G)
        
        #for array in the tuple that comes out of np.nonzero(thin(B))
        # or just one if it's just a single thing i guess?
        
        # the x, y points of the cutmarks are in columns
        cutinds = np.stack(cutmarks)

        for P in cutinds.T:

            rmin, rmax = max(0, P[0]-100), min(img.shape[0], P[0]+100)
            cmin, cmax = max(0, P[1]-100), min(img.shape[1], P[1]+100)
            window = np.s_[rmin:rmax, cmin:cmax]
        
            # perimeter indices within the window
            pinds = [(x,y) for x, y in zip(*perimeter)
                           if x > rmin and x < rmax and y > cmin and y < cmax
                           ]
        
            # shortest distance to boundary point
            r = min(l2_dist(P, pp) for pp in pinds)

            B = np.zeros_like(img.mask)
        
            B[cutmarks] = True
            
            # center a disk of found radius there
            D = disk(r)
            B[P[0]-r :P[0]+r+1 , P[1]-r:P[1]+r+1] = D

            dilcut[B] = ma.masked

       

            montage = np.hstack((gray2rgb(img.filled(0)[window]),
                                gray2rgb(C.filled(0)[window]),
                                ucip[window],
                                gray2rgb(dilcut.filled(0)[window])))
            plt.imshow(montage)
            plt.show()
            plt.close()

print("*"*80)
print(f"there were {len(placentas)} total samples",
      f"and {len(samples_with_cuts)} of them had cuts")
