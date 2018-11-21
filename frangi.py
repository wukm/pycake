import numpy as np
import numpy.ma
from hfft import fft_hessian
from diffgeo import principal_curvatures
from plate_morphology import dilate_boundary


def frangi_from_image(img, sigma, beta=0.5, gamma=None, dark_bg=True,
                      dilation_radius=None, kernel='discrete',
                      signed_frangi=False, return_debug_info=False):
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
    hesh = fft_hessian(img, sigma, kernel=kernel)  # the triple (Hxx,Hxy,Hyy)

    k1, k2 = principal_curvatures(img, sigma, H=hesh)

    if dilation_radius is not None:

        # pass None to just get the mask back
        collar = dilate_boundary(None, radius=dilation_radius,
                                 mask=img.mask)

        # get rid of "bad" K values before you calculate gamma
        k1[collar] = 0
        k2[collar] = 0

    # set default gamma value if not supplies
    if gamma is None:
        gamma = .5 * max_hessian_norm(hesh)
        if np.isclose(gamma, 0):
            print("WARNING: gamma is close to 0. should skip this layer.")

    targets = get_frangi_targets(k1, k2, beta=beta, gamma=gamma,
                                 dark_bg=dark_bg, signed=signed_frangi)

    if not return_debug_info:
        return targets
    else:
        return targets, (R, S, gamma)

def get_frangi_targets(K1, K2, beta=0.5, gamma=None, dark_bg=True,
                       signed=False):
    """Calculate the Frangi vesselness measure from eigenvalues.

    Parameters
    ----------
        K1, K2 : ndarray (each)
            each is an ndarray of eigenvalues (approximated principal
            curvatures) for some image.
        beta: float
            the anisotropy parameter (default is 0.5)
        gamma: float or None
            the structureness parmeter. if gamma is None (default), use
            half of L2 norm of hessian (calculated from K2).  if you want to
            use half of frobenius norm, calculate it outside here.
        dark_bg: boolean or None
            if True, then frangi will select only for bright curvilinear
            features; if False, then Frangi will select only for dark
            curvilinear structures. if None instead of a bool, then curvilinear
            structures of either type will be reported.
        signed: boolean
            if signed is True, the result will be the same as if dark_bg is set
            to None, except that the sign will change to match the desired
            features. See example below.

    Returns
    -------
        F: ndarray, same shape as K1
            the Frangi vesselness measure.
    Examples
    --------
    >>>f1 = get_frangi_targets(K1,K2, dark_bg=True, signed=True)
    >>>f2 = get_frangi_targets(K1,K2, dark_bg=False, signed=True)
    >>>f1 == -f2
    True

    """
    R = anisotropy(K1,K2)
    S = structureness(K1,K2)

    if gamma is None:
        # half of max hessian norm (using L2 norm)
        gamma = .5 * np.abs(K2).max()
        if np.isclose(gamma,0):
            print("warning! gamma is very close to zero."
                  "maybe this layer isn't worth it...")
            print("sigma={:.3f}, gamma={}".format(sigma,gamma))
            print("returning an empty array")
            return np.zeros_like(img)

    F = np.exp(-R / (2*beta**2))
    F *= 1 - np.exp( -S / (2*gamma**2))

    # now just filter/ change sign as appropriate.
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
            # output is already signed
            pass
        elif dark_bg:
            # positive curvature spots will be made negative
            F[K2 > 0] = -1 * F[K2 > 0]
        else:
            # negative curvature spots will be made positive
            F[K2 < 0] = -1 * F[K2 < 0]

    # reapply the mask if the inputs came with one
    if numpy.ma.is_masked(K1):
        F = numpy.ma.masked_array(F, mask=K1.mask)

    return F

def max_hessian_norm(hesh):
    """Calculate max norm of Hessian.
    calculates the maximal value (over all pixels of the image) of the
    Frobenius norm of the Hessian.

    Parameters
    ----------
    hesh: a tuple of ndarrays
        The tuple hxx,hxy,hyy which are all the same shape. The hessian at
        the point (m,n) is then [[hxx[m,n], hxy[m,n]],
                                 [hxy[m,n], hyy[m,n]]]

    Returns
    -------

    """
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
