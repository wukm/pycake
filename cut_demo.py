#!/usr/bin/env python3

"""
this was a method of masking cutmarks by identifying their place in the
original tracing file. the functionality of this program has been eclipsed by
the function find_plate, which attempts to grab the entire perimeter of the
plate, cut and all, and does not require the mask at all.
"""

import numpy as np
import matplotlib.pyplot as plt

from placenta import (open_typefile, list_placentas, get_named_placenta)
from plate_morphology import mask_cuts_simple, dilate_boundary

from skimage.color import gray2rgb
from skimage.morphology import thin, binary_dilation, disk, square
import numpy.ma as ma
import os.path

def l2_dist(p,q):
    return int(np.round(np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)))

placentas = list_placentas('T-BN')
samples_with_cuts = list()

for filename in list_placentas('T-BN'):

    # this one has two cuts, another one has two cuts as well
    #if filename != "T-BN2459820.png":
    #       continue

    img = get_named_placenta(filename)
    ucip = open_typefile(filename, 'ucip')

    #C, has_cut = mask_cuts(img, ucip, return_success=True, in_place=False)

    #if has_cut:

    #    dilcut = img.copy()

    #    print(filename, "has a cut!")
    #    samples_with_cuts.append(filename)

    #    B = np.all(ucip==(0,0,255), axis=-1)
    #    G = np.all(ucip==(0,255,0), axis=-1)

    #    cutmarks = np.nonzero(thin(B))
    #    perimeter = np.nonzero(G)

    #    #for array in the tuple that comes out of np.nonzero(thin(B))
    #    # or just one if it's just a single thing i guess?

    #    # the x, y points of the cutmarks are in columns
    #    cutinds = np.stack(cutmarks)

    #    for P in cutinds.T:

    #        # consider larger and larger window sizes
    #        for W in [100,200,300]:
    #            # consider all perimeter elements within these bounds

    #            rmin, rmax = max(0, P[0]-W), min(img.shape[0], P[0]+W)
    #            cmin, cmax = max(0, P[1]-W), min(img.shape[1], P[1]+W)
    #            window = np.s_[rmin:rmax, cmin:cmax]

    #            # perimeter indices within the window
    #            pinds = [(x,y) for x, y in zip(*perimeter)
    #                        if x > rmin and x < rmax and y > cmin and y < cmax
    #                        ]
    #            if pinds:
    #                break
    #    if pinds:

    #        # max distance to boundary point in the window
    #        # we really only need to keep the largest; deque?
    #        dists = sorted([(pp, l2_dist(P,pp)) for pp in pinds],
    #                        key=lambda t: t[1])
    #        r = int(dists[-1][1]) + 1 # get largest radius but closest point
    #        P = dists[0][0]
    #        B = np.zeros_like(img.mask)

    #        B[cutmarks] = True

    #        # center a disk of found radius there
    #        D = disk(r)
    #        winx = max(P[0]-r,0), min(P[0]+r+1,B.shape[0])
    #        winy = max(P[1]-r,0), min(P[1]+r+1,B.shape[1])
    #        try:
    #            B[winx[0]:winx[1] , winy[0]:winy[1]] = D
    #        except ValueError:
    #            # they're out of bounds so it's a size mismatch. fix it
    #            # by starting/ending D index with opposite sign of the initial
    #            # p +/- radius that was out of bounds
    #            # for example P[0]-r was -9 and everything else was fine
    #            # so you just need to set left side to D[9:,:]
    #            # but you should  wrap this up in a function so the three times
    #            # you do it here and the one time in ucip all gets the same
    #            # code
    #            pass
    #        dilcut[B] = ma.masked

    #    else:
    #        # this is probably not going to happen, but just in case no
    #        # nearby perimeter was found, just... give up
    #        pass

    #    rminv, rmaxv = max(0, rmin-W//2), min(img.shape[0], rmax+W//2)
    #    cminv, cmaxv = max(0, cmin-W//2), min(img.shape[1], cmax+W//2)
    #    view = np.s_[rminv:rmaxv, cminv:cmaxv]
    #    montage = np.hstack((gray2rgb(img.filled(0)[view]),
    #                        ucip[view],
    #                        gray2rgb(C.filled(0)[view]),
    #                        gray2rgb(dilcut.filled(0)[view])))
    #    filestub, _ = os.path.splitext(filename)
    #    plt.imsave(f'demo_output/cut_demo/{filestub}_cutopts.png', montage)
    #    #plt.imshow(montage)
    #    plt.show()
    #    plt.close()
    mimg, success = mask_cuts_simple(img, ucip, return_success=True)
    if success:
        montage = np.hstack((img.filled(0),
                             mimg.filled(0)))
        plt.imshow(montage)
        plt.show()
        plt.close()
print("*"*80)
print(f"there were {len(placentas)} total samples",
      f"and {len(samples_with_cuts)} of them had cuts")
