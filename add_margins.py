#!/usr/bin/env python3

from skimage.filters import sobel
from frangi import frangi_from_image
from plate_morphology import dilate_boundary
from skimage.morphology import remove_small_holes, remove_small_objects
from merging import nz_percentile

s = sobel(img)
s = dilate_boundary(s, mask=img.mask, radius=20)
finv = frangi_from_image(s, sigma=0.8, dark_bg=True)

finv_thresh = nz_percentile(finv, 80)

margins = remove_small_objects((finv > ft).filled(0), min_size=32)

margins_added = remove_small_holes(np.logical_or(margins, approx),
                                   min_size=100, connectivity=2)

markers = np.zeros(img.shape, dtype=np.uint8)

markers[ Fmax < .1 ] = 1
markers[ margins_added] = 2

rw = random_walker(img, markers)
approx_rw = (rw==2)
confusion_rw = confusion(approx_rw, trace, bg_mask=ucip_mask)
mccs_rw = mccs(approx_rw, trace, bg_mask=ucip_mask)
pnc_rw = np.logical_and(skeltrace, rw2==2).sum() / skeltrace.sum()
