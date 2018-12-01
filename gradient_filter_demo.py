# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from placenta import get_named_placenta, list_by_quality, cropped_args
from frangi import frangi_from_image
from hfft import fft_gradient
from merging import nz_percentile
from plate_morphology import dilate_boundary
import os.path, os

OUTPUT_DIR = 'demo_output/gradient_filter_demo'

BETA = .5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)

F0 = list()
F1 = list()

scales = np.logspace(-1, 3, num=12, base=2)

for n, sigma in enumerate(scales):

    f0 = frangi_from_image(img, sigma, beta=BETA, dark_bg=False, dilation_radius=20, gradient_filter=False)
    f1 = frangi_from_image(img, sigma, beta=BETA, dark_bg=False, dilation_radius=20, gradient_filter=True)

    # simulate g
    #g = fft_gradient(img, sigma)
    #g = dilate_boundary(g, radius=20, mask=img.mask)
    #g = g < nz_percentile(g, 50)
    #g_filter = (~g).filled(0)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,4))

    ax[0].imshow(f0.filled(0)[crop], vmin=0, vmax=1, cmap='nipy_spectral')
    ax[0].axis('off')
    ax[0].set_title(f'Standard Frangi σ={sigma:.2f}', fontsize=10)

    ax[1].imshow(f1.filled(0)[crop], vmin=0, vmax=1, cmap='nipy_spectral')
    ax[1].axis('off')
    ax[1].set_title(f'w/ gradient filter', fontsize=10)
    
    #ax[2].imshow(g_filter[crop].T, cmap='nipy_spectral')
    #ax[2].axis('off')
    #ax[2].set_title(f'Gradient filter σ={sigma:.2f}')

    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gf_scale_{n:0{2}}.png'))
    plt.close('all')

    F0.append(f0)
    F1.append(f1)

F0 = np.stack(F0)
F1 = np.stack(F1)


fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,4))

ax[0].imshow(F0.max(axis=0).filled(0)[crop], vmin=0, vmax=1,
        cmap='nipy_spectral')
ax[0].axis('off')
ax[0].set_title(f'Standard Frangi F_max', fontsize=10)

ax[1].imshow(F1.max(axis=0).filled(0)[crop], vmin=0, vmax=1,
        cmap='nipy_spectral')
ax[1].axis('off')
ax[1].set_title(f'w/ gradient filter', fontsize=10)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, f'gf_Fmax.png'))
plt.close('all')

