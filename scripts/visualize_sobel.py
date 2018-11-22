
from skimage.filters import sobel
from plate_morphology import dilate_boundary

# this has to be run after you have Fmax (max of Frangi score over scales)
S = sobel(img)
S /= S.max()

# remove large response from boundary
S = dilate_boundary(S, mask=img.mask).filled(0)
# put normalized sobel into an RGB channel
Z = np.zeros_like(S)
comb = np.dstack((Fmax / Fmax.max(),S,Z))
plt.imshow(comb[crop])
