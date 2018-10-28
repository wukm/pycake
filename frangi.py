import numpy as np
import numpy.ma
from hfft import fft_hessian
from diffgeo import principal_curvatures
from plate_morphology import dilate_boundary

def frangi_from_image(img, sigma, beta=0.5, gamma=None, dark_bg=True,
                      dilation_radius=None, threshold=None,
                      return_debug_info=False):
    """
    Perform a frangi filtering on img
    if None, gamma returns half of Frobenius norm on the image
    if dilation radius is specified, that amount is dilated from the
    boundary of the image (mask must be specified)

    input image *must* be a masked array. To implement: supply mask
    or create a dummy mask if not specified so this can work out of the
    box on arbitrary images.

    return_debug info will return anisotropy, structureness measures, as
    well as the calculated gamma. will return a tuple of
    (R, S, gamma) where R and S are matrices of shape img.shape
    and gamma is a float.


    BIGGER TODO:

        THIS OVERLAPS WITH pcsvn.make_multiscale
        USE THIS THERE
    """
    # principal_directions() calculates the frangi filter with
    # standard convolution and takes forever. FIX THIS!
    hesh = fft_hessian(img, sigma) # the triple (Hxx,Hxy,Hyy)

    k1, k2 = principal_curvatures(img, sigma, H=hesh)

    if dilation_radius is not None:

        # pass None to just get the mask back
        collar = dilate_boundary(None, radius=dilation_radius,
                                 mask=img.mask)

        # get rid of "bad" K values before you calculate gamma
        k1[collar] = 0
        k2[collar] = 0

    if gamma is None:

        gamma = .5 * max_hessian_norm(hesh)
        if np.isclose(gamma,0):
            print("WARNING: gamma is close to 0. should skip this layer.")

    targets = get_frangi_targets(k1, k2, beta=beta, gamma=gamma,
                                 dark_bg=dark_bg, threshold=threshold)

    if not return_debug_info:
        return targets
    else:
        return targets, (R, S, gamma)

def get_frangi_targets(K1,K2, beta=0.5, gamma=None, dark_bg=True, signed=False, threshold=None):
    """
    returns results of frangi filter. eigenvalues are inputs

    if gamma is not supplied, use half of L2 norm of hessian
    if you want to use half of frobenius norm, calculate it outside here
    if dark bg is None instead of a bool, then  ignore and don't filter based on
    positive or negative curvature at all (magnitude is important only)
    however, if signed is True, then dark_bg will be completely ignored, and you'll get
    positive where K2 > 0 and negative where K2 < 0 (theoretically you could flip it)
    """

    R = anisotropy(K1,K2)
    S = structureness(K1,K2)

    if gamma is None:
        # half of max hessian norm (using L2 norm)
        gamma = .5 * np.abs(K2).max()
        if np.isclose(gamma,0):
            print("warning! gamma is very close to zero. maybe this layer isn't worth it...")
            print("sigma={:.3f}, gamma={}".format(sigma,gamma))
            print("returning an empty array")
            return np.zeros_like(img)

    F = np.exp(-R / (2*beta**2))
    F *= 1 - np.exp( -S / (2*gamma**2))

    if not signed:
        # calculate the regular frangi filter
        if dark_bg is None:
            #keep F the way it is
            pass
        elif dark_bg:
            # zero responses from positive curvatures
            F = (K2 < 0)*F
        else:
            # zero responses from negative curvatures
            F = (K2 > 0)*F
    else:
        if dark_bg is None:
            # nothing really makes sense here
            pass
        elif dark_bg:
            # positive curvature spots will be made negative
            F[K2 > 0] = -1 * F[K2 > 0]
        else:
            # negative curvature spots will be made positive
            F[K2 < 0] = -1 * F[K2 < 0]

    if numpy.ma.is_masked(K1):
        F = numpy.ma.masked_array(F, mask=K1.mask)
    if threshold:

        return F < threshold
    else:
        return F

def max_hessian_norm(hesh):

    hxx, hxy, hyy = hesh

    # frob norm is just sqrt(trace(AA^T)) which is easy for a 2x2
    max_norm = np.sqrt((hxx**2 + 2*hxy*2 + hyy**2).max())

    return max_norm

def anisotropy(K1,K2):
    """
    according to Frangi (1998) this is technically A**2
    """

    return (K1/K2) **2

def structureness(K1,K2):
    """
    according to Frangi (1998) this is technically S**2
    """
    return K1**2 + K2**2
