#!/usr/bin/env python3

"""
show the scale to argmax demo
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import (get_named_placenta, list_by_quality, cropped_args,
                      mimg_as_float, open_tracefile, open_typefile)

from frangi import frangi_from_image
import numpy.ma as ma
from hfft import fft_gradient, fft_hessian, fft_gaussian
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

from preprocessing import inpaint_hybrid

from scoring import (scale_to_argmax_plot, scale_to_width_plots,
                     merge_widths_from_traces)



filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
img = inpaint_hybrid(img)
crop = cropped_args(img)
trace = open_tracefile(filename)
A_trace = open_typefile(filename, 'arteries')
V_trace = open_typefile(filename, 'veins')
widths = merge_widths_from_traces(A_trace, V_trace, strategy='arteries')


beta=.10; gamma=0.5;
N_scales = 8
log_range=(0.25, 3)
scales = np.logspace(*log_range, base=2, num=N_scales)

V = np.stack([frangi_from_image(img, sigma, beta=beta, gamma=gamma,
                                dark_bg=False, dilation_radius=20,
                                rescale_frangi=True).filled(0)
              for sigma in scales])

Vmax = V.max(axis=0)
Vargmax = V.argmax(axis=0)
cm = plt.cm.nipy_spectral

plt.imshow(Vmax, cmap=cm, vmin=0, vmax=1)


scale_to_width_plots(V, Vargmax, widths, scales, cmap=cm)
