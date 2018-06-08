import numpy as np
import numpy.ma

def get_frangi_targets(K1,K2, beta=0.5, gamma=15, dark_bg=True, threshold=None):
    """
    returns results of frangi filter
    """

    R = (K1/K2) ** 2 # anisotropy
    S = (K1**2 + K2**2) # structureness

    F = np.exp(-R / (2*beta**2))
    F *= 1 - np.exp( -S / (2*gamma**2))

    if dark_bg:
        F = (K2 < 0)*F
    else:
        F = (K2 > 0)*F

    if numpy.ma.is_masked(K1):
        F = numpy.ma.masked_array(F, mask=K1.mask)
    if threshold:

        return F < threshold
    else:
        return F
