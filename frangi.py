import numpy as np
import numpy.ma
from hfft import fft_hessian
from diffgeo import principal_curvatures
from plate_morphology import dilate_boundary


def frangi_from_image(img, sigma, beta=0.5, gamma=0.5, c=None, dark_bg=True,
                      dilation_radius=None, kernel=None, signed_frangi=False,
                      return_debug_info=False, verbose=False):
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
    gamma: float, optional
        Scaling factor for the structureness parameter of the Frangi
        filter. The structureness parameter will be set to gamma * maximum
        of the hessian norm. (Default is 0.5)
    c: float or None, optional
        The strutureness parameter of the Frangi filter. If this is set then
        gamma is ignored. (Default is None).
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
    Returns
    -------
    ...

    Notes
    -----
    Although default is 0.5, this means that the structureness factor of the
    Frangi score will only be 0.86 at its maximum. Larger values of gamma
    will only dampen the frangi filter more. Smaller values toward 0 will
    result in a "looser" filter. For example, if gamma = .25, then the
    maximum score is (1-exp{-8}) around .999 (it may be desirable that the
    franginess score should be able to achieve a score of 1).

    This function will accept 0 an input, and the structureness factor will
    be set to 1 everywhere (the limiting case as gamma -> 0)

    Frangi structureness factor is (1 - exp((-S**2)/(2*c**2))
    """
    hesh = fft_hessian(img, sigma, kernel=kernel)  # the triple (Hxx,Hxy,Hyy)
    # calculate principal curvatures with |k1| <= |k2|

    k1, k2 = principal_curvatures(img, sigma, H=hesh)

    if dilation_radius is not None:
        # pass None to just get the mask back
        collar = dilate_boundary(None, radius=dilation_radius, mask=img.mask)

        # get rid of "bad" K values before you calculate gamma and Frangi
        k1[collar] = 0
        k2[collar] = 0
        hesh[0][collar] = 0
        hesh[1][collar] = 0
        hesh[2][collar] = 0
    else:
        collar = img.mask.copy()

    # no need to set gamma or c anymore. will be set inside get_frangi_targets
    #if c is None:
    # Frangi suggested 'half the max Hessian norm' as an empirical
    # half the max spectral radius is easier to calculate so do that
    # shouldn't be affected by mask data but should make sure the
    # mask is *well* far away from perimeter
    # we actually calculate half of max hessian norm
    # using frob norm = sqrt(trace(AA^T))
    # alternatively you could use gamma = .5 * np.abs(k2).max()
    #hnorm = hessian_norm(hesh, mask=collar)
    #print(f'σ={sigma:2f}')
    #gamma0 = .5*hessian_norm(hesh).max()
    #print(f'\t{gamma0:.5f} = frob-norm γ pre-dilation')

    #gamma1 = .5*hessian_norm(hesh, mask=collar).max()
    #print(f'\t{gamma1:.5f} = frob-norm γ post-collar dilation {dilation_radius}')
    #l2gamma = .5*np.max(np.abs(k2))
    #print(f'\t{l2gamma:.5f} =  from L2-norm γ (K2 with collar)')

    #hdilation = int(max(np.ceil(sigma),10))
    #hcollar = dilate_boundary(None, radius=hdilation, mask=img.mask)
    #gamma = .5 * max_hessian_norm(hesh, mask=hcollar)

    #print(f'\t{gamma:.5f} = γ post-hdilation (radius {hdilation}) (old γ)')

    #print('changing γ to L2-norm with collar')
    #gamma = max(gamma1, l2gamma, gamma, gamma0)

    # wish this scaled a little better

    # a very large gamma here will make the Frangi score zero
    # a very small gamma means that we are artificially inflating the
    # structureness measure
    #import matplotlib.pyplot as plt
    #plt.imshow(hnorm*(~collar))
    #plt.show()
    #print(hnorm[~collar].min(), hnorm[~collar].max())
    #if hnorm[~collar].max() < 0.1:
    #    print(f'max hessian norm is very small at this scale ({sigma},{hnorm[~collar].max():.3f})',
    #            'you should maybe skip this scale')
    #elif hnorm[~collar].min() <  0.001:
    #    # only trigger if the first one didn't
    #    print(f'min hessian norm is very small at this scale ({sigma}, {hnorm[~collar].min():.3f})',
    #          'be carefully of artificially inflated scores')
    #S = np.sqrt(k1**2 + k2**2)
    #import matplotlib.pyplot as plt
    #plt.imshow(S)
    #plt.show()
    #plt.close()
    #print('max hessian norm (Frob): ', hnorm.max())
    #print('max structureness: ', S.max())
    #c = gamma*S.max()

    if verbose:
        print(f'finding Frangi targets with β={beta} and γ={c:.2}')

    targets = get_frangi_targets(k1, k2, beta=beta, gamma=gamma, c=c,
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


def get_frangi_targets(K1, K2, beta=0.5, gamma=0.5, c=None,
                       dark_bg=True, signed=False):
    """Calculate the Frangi vesselness measure from eigenvalues.

    Parameters
    ----------
        K1, K2 : ndarray (each)
            each is an ndarray of eigenvalues (approximated principal
            curvatures) for some image.
        beta: float
            the anisotropy parameter (default is 0.5)
        gamma: float or None
            Scaling factor for the the structureness parmeter. The structurness
            parameter c will be set to gamma times the maximum of the Hessian
            norm, sqrt(K1**2 + K2**2). Default is 0.5
        c: float or None
            The frangi structurness parameter. If this is set, gamma above will
            be ignored.
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

    Notes
    -----
    If beta or gamma are set to 0, then the frangi anisotropy factor will be
    set to 0 or 1 everywhere (which is the limiting case as beta->0 or
    gamma->0) You can set beta = 'inf' or np.inf to set anisotropy factor to 1.

    Examples
    --------
    >>>f1 = get_frangi_targets(K1,K2, dark_bg=True, signed=True)
    >>>f2 = get_frangi_targets(K1,K2, dark_bg=False, signed=True)
    >>>np.all(f1 == -f2)
    True

    >>> F = get_frangi_targets(K1,K2, gamma=0.5)
    >>> Falt = get_frangi_targets(K1,K2, c=0.5*np.sqrt(K1**2 + K2**2))
    >>> np.all(F == Falt)
    True
    """

    A = anisotropy(K1, K2, beta=beta)
    S = structureness(K1, K2, gamma=gamma, c=c)

    anisotropy_factor = np.exp(-A)
    structureness_factor = (1 - np.exp(-S))

    F = anisotropy_factor * structureness_factor

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

    # finally, reapply the mask if the inputs came with one
    if numpy.ma.is_masked(K1):
        F = numpy.ma.masked_array(F, mask=K1.mask)

    return F


def hessian_norm(hesh, mask=None):
    """Calculate Frobenius norm of Hessian.

    Calculates the maximal value (over all pixels of the image) of the
    Frobenius norm of the Hessian. This should be the same as the square root
    of unscaled structureness.

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
    return hnorm


def anisotropy(K1,K2, beta=0.5):
    """Convenience function for the exponential argument in the Frangi
    anisotropy factor.

    According to Frangi (1998) this is technically (A**2) / (2*beta**2)
    unless beta is None, in which case just A**2 is returned

    The frangi vesselness factor is formally (np.exp(-R))
    where R is what's returned by this function
    """

    A = (K1 / K2) ** 2

    if beta == 0:
        return np.zeros_like(A)  # the limiting case as beta -> 0

    elif beta == 'inf' or np.isinf(beta):
        return np.ones_like(A)  # the limiting case as beta -> inf

    elif beta is None:
        return A  # just return the A**2 part (why though)
    else:
        return A / (2*beta**2)


def structureness(K1, K2, gamma=0.5, c=None):
    """Convenience function for Structureness measure.
    According to Frangi (1998) this is technically S**2
    """
    S = K1**2 + K2**2

    # is c is not provided, calculate it
    if c is None:
        c = gamma * S.max()  # the max Frob norm of the Hessian

    if c == 0:
        return np.zeros_like(S)

    elif c == 'inf' or np.isinf(c):
        return np.ones_like(S)

    elif c is None:
        return S

    else:
        return S / (2*c**2)
