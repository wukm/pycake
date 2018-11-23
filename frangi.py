import numpy as np
import numpy.ma
from hfft import fft_hessian
from diffgeo import principal_curvatures
from plate_morphology import dilate_boundary


def frangi_from_image(img, sigma, beta=0.5, gamma=None, dark_bg=True,
                      dilation_radius=None, kernel=None,
                      signed_frangi=False, return_debug_info=False,
                      verbose=False):
    """Calculate the (uniscale) Frangi vesselness measure on a grayscale image

    Parameters
    ----------
    img: ndarray or ma.MaskedArray
        a one-channel image. If this is a masked array (preferred), ignore the
        masked regions of the image
    sigma: float
        Standard deviation of the gaussian, used to calculate derivatives.
    beta: float, optional
        The anisotropy parameter of the Frangi filter (default is 0.5)
    gamma: float or None, optional
        The structureness parameter of the Frangi filter. If None, gamma
        returns half of Frobenius norm of the calculated hessian (Default is
        None).
    dilation_radius: int or None
        If dilation radius is supplied, then areas within that amount of pixels
        will not be calculated. This is preferable in certain contexts,
        especially when there is a dark background and dark_bg=True. This is
        especially recommended for small sigmas and when gamma is not provided.
        None to forgo this procedure (default).  A mask must be supplied for
        this to make sense.
    dark_bg: boolean or None
        if True, then frangi will select only for bright curvilinear
        features; if False, then Frangi will select only for dark
        curvilinear structures. if None instead of a bool, then curvilinear
        structures of either type will be reported.
    signed_frangi: bool, optional
        if signed is True, the result will be the same as if dark_bg is set
        to None, except that the sign will change to match the desired
        features. See example below.
    return_debug info: bool, optional
        will return a large dict consisting of several large matrices,
        calculated hessian, etc.

        scale_dict = {'sigma': sigma,
                      'beta': beta,
                      'gamma': gamma,
                      'H': hesh,
                      'F': targets,
                      'k1': k1,
                      'k2': k2,
                      'border_radius': dilation_radius
                      }

    """
    hesh = fft_hessian(img, sigma, kernel=kernel)  # the triple (Hxx,Hxy,Hyy)
    # calculate principal curvatures with |k1| <= |k2|

    k1, k2 = principal_curvatures(img, sigma, H=hesh)

    if dilation_radius is not None:
        # pass None to just get the mask back
        collar = dilate_boundary(None, radius=dilation_radius, mask=img.mask)

        # get rid of "bad" K values before you calculate gamma
        k1[collar] = 0
        k2[collar] = 0

    # set default gamma value if not supplied
    if gamma is None:
        # Frangi suggested 'half the max Hessian norm' as an empirical
        # half the max spectral radius is easier to calculate so do that
        # shouldn't be affected by mask data but should make sure the
        # mask is *well* far away from perimeter
        # we actually calculate half of max hessian norm
        # using frob norm = sqrt(trace(AA^T))
        # alternatively you could use gamma = .5 * np.abs(k2).max()
        print(f'σ={sigma:2f}')
        gamma0 = .5*max_hessian_norm(hesh)
        print(f'\t{gamma0:.5f} = frob-norm γ pre-dilation')

        gamma1 = .5*max_hessian_norm(hesh, mask=collar)
        print(f'\t{gamma1:.5f} = frob-norm γ post-collar dilation {collar_radius}')

        l2gamma = .5*np.max(np.abs(k2))
        print(f'\t{l2gamma:.5f} =  from L2-norm γ (K2 with collar)')

        hdilation = int(max(np.ceil(sigma),10))
        hcollar = dilate_boundary(None, radius=hdilation, mask=img.mask)
        gamma = .5 * max_hessian_norm(hesh, mask=hcollar)
        print(f'\t{gamma:.5f} = γ post-hdilation (radius {hdilation}) (old γ)')

        print('changing γ to L2-norm with collar')
        gamma = l2gamma
        print('-'*80)

        if verbose:
            print(f"gamma (half of max hessian (frob) norm is {gamma}")

        # make a better test?
        if np.isclose(gamma, 0):
            print("WARNING: gamma is close to 0. should skip this layer.")

    if verbose:
        print(f'finding Frangi targets with β={beta} and γ={gamma:.2}')

    targets = get_frangi_targets(k1, k2, beta=beta, gamma=gamma,
                                 dark_bg=dark_bg, signed=signed_frangi)

    if not return_debug_info:
        return targets
    else:
        scale_dict = {'sigma': sigma,
                      'beta': beta,
                      'gamma': gamma,
                      'H': hesh,
                      'F': targets,
                      'k1': k1,
                      'k2': k2,
                      'border_radius': dilation_radius
                      }

        return targets, scale_dict


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

    if gamma is None:
        # half of max hessian norm (using L2 norm)
        gamma = .5 * np.abs(K2).max()
        if np.isclose(gamma, 0):
            print("warning! gamma is very close to zero."
                  "maybe this layer isn't worth it...")
            print("returning an empty array")

            return np.zeros_like(K1)

    R = anisotropy(K1, K2, beta=beta)
    S = structureness(K1, K2, gamma=gamma)

    F = np.exp(-R)
    F *= 1 - np.exp(-S)

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

def max_hessian_norm(hesh, mask=None):
    """Calculate max Frobenius norm of Hessian.

    Calculates the maximal value (over all pixels of the image) of the
    Frobenius norm of the Hessian.

    Parameters
    ----------
    hesh: a tuple of ndarrays
        The tuple hxx,hxy,hyy which are all the same shape. The hessian at
        the point (m,n) is then [[hxx[m,n], hxy[m,n]],
                                 [hxy[m,n], hyy[m,n]]]

    Returns
    -------
    float
    """

    hxx, hxy, hyy = hesh

    # frob norm is just sqrt(trace(AA^T)) which is easy for a 2x2
    hnorm = (hxx**2 + 2*hxy**2 + hyy**2)

    if mask is not None:
        hnorm[mask] = 0

    hnorm = np.sqrt(hnorm)
    return hnorm.max()


def anisotropy(K1,K2, beta=None):
    """Convenience function for Anisotropy measure.

    According to Frangi (1998) this is technically A**2
    """

    A = (K1/K2) **2

    if beta is None:
        return A
    else:
        return A / (2*beta**2)

def structureness(K1,K2, gamma=None):
    """Convenience function for Structureness measure.
    According to Frangi (1998) this is technically S**2
    """
    S = K1**2 + K2**2

    if gamma is None:
        return S
    else:
        return S / (2*gamma**2)
