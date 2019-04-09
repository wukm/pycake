#!/usr/bin/env python3

"""
This is a visual demonstration of the Frangi filter as a function.
The output is a sequence of 3D graphs, varying the parameters beta and gamma
"""

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from skimage.io import imread
from skimage.util import montage
from itertools import product
import seaborn as sns
def s(x, gamma):
    """normalized structureness factor.
    x is the ratio of the input to the maximum possible structureness factor
    (smax) at that scale (which would cancel out with the c in the denominator)
    """
    return (1 - exp(-x**2 / (2*gamma**2))) / (1 - exp(-1/(2*gamma**2)))


def r(y, beta):
    """normalized anistropy factor
    y is the ratio |k1 / k2|, so y=0 corresponds to perfectly isotropic
    and y->1 corresponds to highly anisotropic
    """
    return np.exp(-y**2 / (2*beta**2))

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


dom = np.linspace(0, 1)

plt.close('all')
#sns.set()
# show dependence of structureness factor on its parameter
for gamma in [0.1, 0.25, 0.35, 0.5, 0.9, 1, 2, 10, 1000]:
    plt.plot(dom, s(dom, gamma), label=r'$\gamma={}$'.format(gamma))

plt.ylabel(r'$\left(1-\exp\left\{\frac{-S}{2(\gamma S_{max})^2}\right\}\right)$',
           fontsize=24)
plt.xlabel(r'$(S / S_{max})$', fontsize=24)
plt.title(r'Dependence of Structureness Factor on Parameter $\gamma=(c/S_{max})$')
plt.legend()

#plt.show()
plt.close('all')

for beta in [0.1, 0.25, 0.35, 0.5, 0.9, 1, 2, 10, 1000]:
    plt.plot(dom, r(dom, beta), label=r'$\beta={}$'.format(beta))

plt.ylabel(r'$\exp\left\{ \frac{-A}{2\beta^2}\right \}$', fontsize=24)
plt.xlabel(r'$ A = \left|\lambda_1 / \lambda_2\right|$', fontsize=24)
plt.title(r'Dependence of Anisotropy Factor on Parameter $\beta$')
plt.legend()

#plt.show()
plt.close('all')

prange = [0.1, 0.25, 0.5, 0.9, 1, 1.5]

for n, (beta, gamma) in enumerate(product(prange, prange)):

    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(dom, dom)

    Z = r(X,beta)*s(Y,gamma)

    surf = ax.plot_surface(X,Y,Z, cmap='coolwarm', linewidth=0)

    ax.set_xlabel(r'$\left|\lambda_1 / \lambda_2 \right|$', fontsize=18)
    ax.set_ylabel(r'$ ( S / S_{\max} )$ ', fontsize=18)

    ax.set_title(r"Rescaled Frangi filter, "
                    r"$\beta={}, \gamma={}$".format(beta,gamma), fontsize=18)

    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()

    plt.savefig(f'demo_output/frangi3d-alt/{n}.png', dpi=300)

    plt.close()

imgs = [imread(f'demo_output/frangi3d-alt/{n}.png') for n in range(36*1)]
imgs = np.stack(imgs)

for n in range(6):
    plt.imsave(f'demo_output/frangi3d-alt/frangi3dpart{n}.png',
               montage(imgs[(n*6):((n+1)*6)], multichannel=True,
                       grid_shape=(3,2)))
