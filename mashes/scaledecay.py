# coding: utf-8
from placenta import get_named_placenta, list_by_quality
list_by_quality(0)
filename = _[-2]
img = get_named_placenta(filename)
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(img)
pls.show()
plt.show()
filename =  list_by_quality(0)[0]
img = get_named_placenta(filename)
from placenta import cropped_args
crop = cropped_args(img)
img[crop]
plt.imshow(img[crop])
plt.show()
from hfft import fft_gaussian
get_ipython().run_line_magic('pinfo', 'fft_gaussian')
C = fft_gaussian(img, 32, 'discrete')
B = fft_gaussian(img, 5, 'discrete')
A = fft_gaussian(img, .12, 'discrete')
plt.imshow(A)
plt.show()
A = fft_gaussian(img, .25, 'discrete')
plt.imshow(A)
plt.show()
plt.imshow(B)
plt.show()
plt.imshow(C)
plt.show()
gA = np.gradient(A)
gA
gB = np.gradient(B)
gC = np.gradient(C)
gA.shape
gA[0].shape
aa = lambda g: np.sqrt((1+g[0]*g[0] + g[0]*g[0]))
plt.imshow(aa(gA))
plt.show()
from plate_morphology import dilate_boundary
from functools import partial
dilate = partial(dilate_boundary, mask=img.mask)
dilate(aa(gA), 20)
plt.imshow(_)
plt.show()
dilate(aa(gA), 20).filled(0)
plt.show()
plt.imshow(_)
plt.show()
Aaa = dilate(aa(gA), 20).filled(0)
Aaa.max()
Aaa.min()
Aaa[~img.mask].min()
Aaa[img.mask].min()
Aaa[img.mask].max()
Aaa[~img.mask].max()
Baa = dilate(aa(gB), 20).filled(0)
Caa = dilate(aa(gC), 20).filled(0)
plt.imshow(Baa)
plt.show()
plt.imshow(Caa)
plt.show()
Caa.min()
Caa[~img.mask].min()
Caa == 0
plt.imshow(_)
plt.show()
Caa[~img.mask].max()
Caa[~img.mask].min()
Caa[~img.mask].argmin()
dilate_boundary(img.mask, 20)
dilate_boundary(img.mask, 20, mask_only=True)
get_ipython().run_line_magic('pinfo', 'dilate_boundary')
dilate_boundary(img, radius=20)
dil = _.mask.copy()
dil
plt.imshow(dil)
plt.show()
Caa[~dil].argmin()
Caa[~dil]
Caa[~dil].min()
Caa[~dil].max()
Baa[~dil].max()
Baa[~dil].min()
Aaa[~dil].min()
Aaa[~dil].max()
plt.imshow(Caa)
plt.show()
plt.imshow(Aaa)
plt.show()
#for sigma in np.logspace(-3, 8, base=2, num=20):
#    D = fft_gaussian(img, sigma, 'discrete')
#    gD = np.gradient(D)
#    Daa = dilate(aa(dG), 20).filled(1)
#    print(Daa[~dil].min(), Daa[~dil].max()
aas = list()
for sigma in np.logspace(-3, 8, base=2, num=20):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa = dilate(aa(dG), 20).filled(1)
    aas.append(Daa)
    print(f"sigma={sigma:.3f}", "min: {:6f}, max:{:6f}".format(
          Daa[~dil].min(), Daa[~dil].max())
          )
          
for sigma in np.logspace(-3, 8, base=2, num=20):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa = dilate(aa(gD), 20).filled(1)
    aas.append(Daa)
    print(f"sigma={sigma:.3f}", "min: {:6f}, max:{:6f}".format(
          Daa[~dil].min(), Daa[~dil].max())
          )
                    
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa = dilate(aa(gD), 20).filled(1)
    aas.append(Daa)
    print(f"sigma={sigma:.3f}", "min: {:6f}, max:{:6f}".format(
          Daa[~dil].min(), Daa[~dil].max())
          )
                    
bb lambda g: np.linalg.norm(np.array(
        [[1+g[1]**2, -g[0]*g[1]],
         [-g[0]*g[1], 1 + g[0]**2]]))
bb = lambda g: np.linalg.norm(np.array(
        [[1+g[1]**2, -g[0]*g[1]],
         [-g[0]*g[1], 1 + g[0]**2]]))
bb(gA)
get_ipython().set_next_input('man np.linalg.norm');get_ipython().run_line_magic('pinfo', 'np.linalg.norm')
ginv = lambda g: np.array(
        [[1+g[1]**2, -g[0]*g[1]],
         [-g[0]*g[1], 1 + g[0]**2]])
ginv(gA)
_.shape
G[:,:,34,289]
ginvA = _101
ginvA[:,:,340,289]
bb = lambda g: np.sqrt((1+g[1]**2)**2 + 2*(g[0]*g[1])**2 + (1+g[0]**2)**2)
bb(gA)
_.shape
plt.imshow(bb)
_.shape
bb(gA)
plt.imshow(_)
plt.show()
bb(gA) / aa(gA)
plt.imshow(_)
plt.show()
dilate(bb(gA) / aa(gA))
plt.imshow(dilate(bb(gA) / aa(gA)).filled(0))
plt.show()
bb = lambda g: np.sqrt((1+g[1]**2)**2 + 2*(g[0]*g[1])**2 + (1+g[0]**2)**2)
plt.imshow(dilate(bb(gA),20))
plt.show()
bb(gA)[~dil].min()
bb(gA)[~dil].max()
(bb(gA) / aa(gA))[~dil].min()
(bb(gA) / aa(gA))[~dil].max()
(bb(gA) / aa(gA))
plt.imshow(dilate(_,20).filled(1.414))
plt.show()
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Daa / Dbb
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(ccs), sep='\n')
           
                    
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Daa / Dbb
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(*aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*ccs), sep='\n')
           
           
                    
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Dbb / Daa
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(*aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*ccs), sep='\n')
                            
data = _
data = list()
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Dbb / Daa
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(*aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*ccs), sep='\n')
    data.append(
        [sigma, *aas, *bbs, *ccs])
         
import pandas
table = pandas.DataFrame(data)
print(table)
from hfft import fft_hessian
get_ipython().run_line_magic('pinfo', 'fft_hessian')
helems = fft_hessian(img, sigma=1, kernel='discrete')
np.sqrt(helems[0]**2 + 2*helems[1]**2 + helems[2]**2)
_.shape
plt.imshow(np.sqrt(helems[0]**2 + 2*helems[1]**2 + helems[2]**2))
plt.show()
data = list()
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Dbb / Daa
    h = fft_hessian(img,sigma, 'discrete')
    hnorm = np.sqrt(h[0]**2 + 2*h[1]**2 + h[2]**2)
    Lnorm = hnorm*Dcc
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    dds = hnorm[~dil].min(), hnorm[~dil].max()
    lls = Lnorm[~dil].min(), Lnorm[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(*aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*ccs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*dds),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*lls), sep='\n')
    data.append(
        [sigma, *aas, *bbs, *ccs, *dds, *lls])
        
         
data
table = pandas.DataFrame(data)
table
table.columns
help(table.columns)
plt.imshow(Lnorm)
plt.show()
Ls = list()
for sigma in np.logspace(-4, 8, base=2, num=50):
    D = fft_gaussian(img, sigma, 'discrete')
    gD = np.gradient(D)
    Daa, Dbb = aa(gD), bb(gD)
    Dcc = Dbb / Daa
    h = fft_hessian(img,sigma, 'discrete')
    hnorm = np.sqrt(h[0]**2 + 2*h[1]**2 + h[2]**2)
    Lnorm = hnorm*Dcc
    Ls.append(Lnorm)
    aas = Daa[~dil].min(), Daa[~dil].max()
    bbs = Dbb[~dil].min(), Dbb[~dil].max()
    ccs = Dcc[~dil].min(), Dcc[~dil].max()
    dds = hnorm[~dil].min(), hnorm[~dil].max()
    lls = Lnorm[~dil].min(), Lnorm[~dil].max()
    print(f"sigma={sigma:.3f}",
           "\tmin: {:.6f},\tmax:{:.6f}".format(*aas),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*bbs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*ccs),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*dds),
           "\tmin: {:.6f},\tmax:{:.6f}".format(*lls), sep='\n')
    #data.append(
    #    [sigma, *aas, *bbs, *ccs, *dds, *lls])
             
L[0]
Ls[0]
plt.imshow(_)
plt.show()
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    plt.imshow(Lnorm[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    L = dilate(Lnorm, min(20,int(sigma))).filled(0)
    plt.imshow(Lnorm[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    L = dilate(Lnorm, min(20,int(sigma))).filled(0)
    plt.imshow(L[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    L = dilate(Lnorm, max(20,int(sigma))).filled(0)
    plt.imshow(L[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    L = dilate(Lnorm, max(20,int(sigma))).filled(0)
    plt.imshow(L[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
for Lnorm, sigma in zip(Ls, np.logspace(-4,8, base=2,num=50)):
    L = dilate(Lnorm, max(20,int(2*sigma))).filled(0)
    plt.imshow(L[crop], cmap='nipy_spectral')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.colorbar()
    plt.title(r'Lnorm $\sigma={:.3f}$'.format(sigma))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
print(table)
table
table + 2
table[:,0]
table[0]
table / table[0]
T = np.array(table)
T
T.shape
T.dtype
T / T[:,0]
T / T[...,0]
T[...,-1] / T[...,0]
T[...,-1] / np.sqrt(T[...,0])
T[...,-1] * np.sqrt(T[...,0])
T[...,-1] * T[...,0]
T[...,-1] * T[...,0]**2
table
T[...,-3] * T[...,0]**2
T[...,-3] * T[...,0]
T[...,-4] * T[...,0]
T[...,-4] / T[...,0]
T[...,-4] / T[...,0] > .01
which = _
T[...,0][which]
(T[...,-4] / T[...,0]) > .001
scales
scales = np.logspace(-4,8,num=50,base=2)
scales
(T[...,-3] / T[...,0]) > .001
(T[...,-3] / T[...,0]) > .005
scales[_]
(T[...,-3] / T[...,0]) > .001
scales
scales
(T[...,-3] / T[...,0]) > .001
scales[_]
(T[...,-3] / T[...,0]) > .05
T[...,-3]
(T[...,-3] / T[...,0])
(T[...,-3] / T[...,0]**2)
(T[...,-3] *np.sqrt(1/T[...,0]))
(T[...,-4] / T[...,0]) > .05
table
from frangi import frangi_from_image
g[0]g[1]
g[0]*g[1]
gA[0]*gA[1]
plt.imshow(_)
plt.show()
