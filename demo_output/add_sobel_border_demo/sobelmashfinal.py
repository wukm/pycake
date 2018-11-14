#!/usr/bin/env python 3


# run extract_NCS_pcsvn.py for just the last sample (T-BN9937513.png)

import numpy as np
from skimage.filters import sobel
from plate_morphology import dilate_boundary
from skimage.morphology import binary_closing

# get sobel image of border
S = sobel(img)
dilate_boundary(S, radius=10, mask=img.mask)
S = dilate_boundary(S, radius=20, mask=img.mask)
Sthresh = (S > nz_percentile(S,90))

# focus in on this smaller region
window = np.s_[130:630,70:570]

# overlay these in different color channels (fix this in post).
plt.imsave('sobel_border_view.png',np.dstack((((F>alphas)*F).max(axis=-1),
                                              0.2*ucip_mask,
                                              0.5*Sthresh.filled(0))
                                             )[crop][window], vmin=0,vmax=0.6)
# show what the trace looks like there
ctrace = open_typefile(filename, 'ctrace')
plt.imsave('ctrace_view.png', ctrace[crop][window])

# and confusion matrix (of standard approximation)
plt.imsave('confusion_view.png', confuse[crop][window])

# and the raw image
plt.imsave('img_view.png', img[crop][window].filled(0), cmap=plt.cm.gray)

# get negative frangi arguments
Fneg, _ = extract_pcsvn(filename, DARK_BG=True, alphas=None, betas=betas, scales=scales, gammas=None, kernel='discrete', dilate_per_scale=True, generate_json=False, output_dir='.')
# calculate percentiles and apply then
negalphas = np.array([nz_percentile(Fneg[:,:,k],95.0) for k in range(n_scales)])
negapprox, _ = apply_threshold(Fneg, negalphas)

# get the largest responses from small scales only, compare to trace
plt.imsave('fneg_confusion_view.png',
           confusion((Fneg > negalphas)[:,:,:20].any(axis=-1), trace,
                     bg_mask=ucip_mask)[crop][window])
