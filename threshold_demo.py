#!/usr/bin/env python3

import numpy as np
if __name__ == "__main__":



    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from skimage.util import img_as_float
    from skimage.io import imread
    from placenta import (get_named_placenta, list_by_quality, cropped_args,
                        mimg_as_float)

    from frangi import frangi_from_image
    import numpy.ma as ma
    from hfft import fft_gradient, fft_hessian, fft_gaussian
    from merging import nz_percentile
    from plate_morphology import dilate_boundary
    import os.path, os
    from skimage.morphology import thin
    from postprocessing import dilate_to_rim
    from scoring import confusion, mcc
    from placenta import open_tracefile, open_typefile  
    from preprocessing import inpaint_hybrid
    from placenta import measure_ncs_markings, add_ucip_to_mask
    
    import json

    mccs = list()
    precs = list()

    scales = np.logspace(-1.5, 3.5, num=20, base=2)
    threshold = .15
    neg_threshold = 0.001
    max_pos_scale = -6
    max_neg_scale = 2
    beta = 0.10
    gamma = 1.0

    cm = mpl.cm.plasma
    #cmscales = mpl.cm.magma
    cm.set_bad('k', 1)  # masked areas are black, not white
    #cmscales.set_bad('w', 1)
    
    QUALITY = 3
    for filename in list_by_quality(QUALITY):

        cimg = open_typefile(filename, 'raw')
        ctrace = open_typefile(filename, 'ctrace')
        trace = open_tracefile(filename)
        img = get_named_placenta(filename)
        crop = cropped_args(img)
        ucip = open_typefile(filename, 'ucip')
        img = inpaint_hybrid(img)
        
        # make the size of figures more consistent
        if img[crop].shape[0] > img[crop].shape[1]:
            # and rotating it would be fix all this automatically
            cimg = np.rot90(cimg)
            ctrace = np.rot90(ctrace)
            trace = np.rot90(trace)
            img = np.rot90(img)
            ucip = np.rot90(ucip)
            crop = cropped_args(img)

        ucip_midpoint, resolution = measure_ncs_markings(ucip)
        ucip_mask = add_ucip_to_mask(ucip_midpoint, radius=60, mask=img.mask)

        name_stub = filename.rstrip('.png').strip('T-')


        F = np.stack([frangi_from_image(img, sigma, beta=beta, gamma=gamma, dark_bg=False,
                                        signed_frangi=True, dilation_radius=20,
                                        rescale_frangi=True)
                                        for sigma in scales])
        # need to fix this in the signed_frangi logic
        F = F*~dilate_boundary(None, mask=img.mask, radius=20)
        
        Fmax = (F*(F>0))[:max_pos_scale].max(axis=0)
        Fneg = -(F*(F<0))[:max_neg_scale].min(axis=0)

        approx = Fmax > threshold
        rim_approx = (Fneg > neg_threshold)
        skel = thin(approx)
        completed = connect_iterative_by_label(skel, Fmax, max_dist=100) 
        completed_dilated = dilate_to_rim(completed, rim_approx, max_radius=10)
        approx_dilated = dilate_to_rim(approx, rim_approx, max_radius=10)
        
        network = np.maximum(skel*3., (completed & ~skel)*2) 
        network = np.maximum(network, rim_approx*1.)


        precision = lambda t: int(t[0]) / int(t[0] + t[2])

        mcc_FA, counts_FA = mcc(approx, trace, bg_mask=img.mask, return_counts=True)
        mcc_FAD, counts_FAD = mcc(approx_dilated, trace, bg_mask=img.mask, return_counts=True)
        mcc_NCD, counts_NCD = mcc(completed_dilated, trace, bg_mask=img.mask, return_counts=True)

        prec_FA = precision(counts_FA)
        prec_FAD = precision(counts_FAD)
        prec_NCD = precision(counts_NCD)

        mccs.append((name_stub, mcc_FA, mcc_FAD, mcc_NCD))
        precs.append((name_stub, prec_FA, prec_FAD, prec_NCD))

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,12))
        A = ax.ravel()

        A[0].imshow(cimg[crop])
        A[0].set_title(name_stub)

        A[1].imshow(ma.masked_where(Fmax==0,Fmax)[crop], cmap=cm, vmin=0, vmax=1)
        A[1].set_title(rf'$V_{{\max}}, \beta={beta:.2f}, \gamma={gamma:.3f}$')
        
        A[2].imshow(network[crop], cmap=plt.cm.magma)
        A[2].set_title('skeleton, completed network, and rim_approx')
        
        A[3].imshow(confusion(approx, trace, bg_mask=img.mask)[crop])
        A[3].set_title(fr'    fixed $\alpha={threshold:.2f}$', loc='left')
        A[3].set_title(f'MCC: {mcc_FA:.2f}\n'
                    f'precision: {prec_FA:.2%}', loc='right')

        A[4].imshow(confusion(approx_dilated, trace, bg_mask=img.mask)[crop])
        A[4].set_title(fr'    dilate_to_rim $\alpha={threshold:.2f}$', loc='left')
        A[4].set_title(f'MCC: {mcc_FAD:.2f}\n'
                    f'precision: {prec_FAD:.2%}', loc='right')

        A[5].imshow(confusion(completed_dilated, trace, bg_mask=img.mask)[crop])
        A[5].set_title(fr'    network_completed, dilated', loc='left')
        A[5].set_title(f'MCC: {mcc_NCD:.2f}\n'
                    f'precision: {prec_NCD:.2%}', loc='right')

        [a.axis('off') for a in A]
        fig.tight_layout()
        plt.show()



runlog = { 'mccs':mccs,
           'precs':precs,
           'scales':list(scales),
           'beta':beta,
           'gamma':gamma,
           'max_pos_scale':max_pos_scale,
           'max_neg_scale':max_neg_scale,
           'threshold':threshold,
           'neg_threshold':neg_threshold
           }


with open(f'output/network_completion_{QUALITY}.json', 'w') as f:
    json.dump(runlog, f, indent=True)
