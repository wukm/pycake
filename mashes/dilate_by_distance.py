# coding: utf-8
F = np.stack((frangi_from_image(img, sigma, dark_bg=True, signed=True, beta=0.35, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5,3.5,num=12,base=2)))
F = np.stack((frangi_from_image(img, sigma, dark_bg=True, signed_frangi=True, beta=0.35, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5,3.5,num=12,base=2)))
F.shape
from merging import view_slices
view_slices(F, axis=0, cmap='seismic', vmin=0, vmax=1)
view_slices(F.filled(0), axis=0, cmap='spectral', vmin=0, vmax=1)
view_slices(F.filled(0), axis=0, cmap='Spectral', vmin=0, vmax=1)
view_slices(np.transpose(F.filled(0),axes=(1,2,0))[crop], axis=-1, cmap='Spectral', vmin=-1, vmax=1)
F.max(axis=0)
plt.imshow(_[crop])
plt.show()
F.min(axis=0)
plt.imshow(_[crop])
plt.show()
F = -F
plt.imshow(F.max(axis=0))
plt.show()
plt.imshow(F.min(axis=0))
plt.show()
fn = (-f).max(axis=0)
fn = (-F).max(axis=0)
fp = (F).max(axis=0)
plt.imshow(_)
plt.imshow(fp)
plt.show()
plt.imshow(fn[crop])
plt.show()
from skimage.morphology import thin
thin(fn)
plt.imshow(_)
plt.show()
thin(fn > .4)
plt.imshow(_)
plt.show()
plt.imshow(fn)
plt.show()
fn.filled(0)
pt.imshow(_)
plt.show()
plt.imshow(_)
plt.show()
fn
fn[img.mask] = ma.masked
import numpy.ma as ma
fn[img.mask] = ma.masked
plt.imshow(fn.filled(0))
plt.show()
plt.imshow(dilate_boundary(fn,radius=20))
plt.show()
plt.imshow(dilate_boundary(fn,radius=25).filled(0))
plt.show()
plt.imshow(dilate_boundary(fn,radius=30).filled(0))
plt.show()
plt.imshow(dilate_boundary(fn,radius=30).filled(0))
fn = _
fp
fp[img.mask] = ma.masked
plt.imshow(fp)
plt.show()
fp = dilate_boundary(fp,radius=30).filled(0)
plt.imshow(fp)
plt.show()
F
thin(fp)
FN = np.stack((frangi_from_image(img, sigma, dark_bg=True, beta=0.35, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5,3.5,num=12,base=2)))
FP = np.stack((frangi_from_image(img, sigma, dark_bg=False, beta=0.35, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5,3.5,num=12,base=2)))
plt.imshow(FN.max(axis=-1))
plt.show()
plt.imshow(FN.max(axis=0))
plt.show()
plt.imshow(FN.max(axis=0))
plt.show()
thin(FP.max(axis=0) > .3)
plt.imshow(_)
plt.show()
from scipy.ndimage import distance_transform_edt
dmarks = np.zeros(img.shape, np.int32)
T = thin(FP.max(axis=0) > .3)
dmarks[:] = 1
dmarks
plt.imshow(FN.max(axis=0) > .3)
plt.show()
plt.imshow(FN.min(axis=0) > .3)
plt.show()
plt.imshow(FN.min(axis=0) < -.3)
plt.show()
plt.imshow(FN)
plt.imshow(FN)
plt.imshow(FN.max(axis=0))
plt.show()
FN.max(axis=0)
FN.max()
FN.max(axis=0) > .3
plt.show()
plt.imshow(_)
plt.shwo()
plt.show()
FN.max(axis=0) > .5
plt.imshow(_)
plt.show()
plt.imshow(FN.max(axis=0))
plt.show()
plt.imshow(FN==0)
plt.imshow(FN.max(axis=0)==0)
plt.show()
plt.imshow(FN.max(axis=0) > .2)
plt.show()
plt.imshow(sobel(FN.max(axis=0)))
from skimage.filters import sobel
plt.imshow(sobel(FN.max(axis=0)))
plt.show()
plt.imshow(sobel(FN.max(axis=0)))
fft_gradient(img,0.4)
plt.imshow(_)
plt.shwo()
plt.show()
fft_gradient(img,0.4)
fft_gradient(img,0.4)
plt.imshow(_)
plt.show()
fft_gradient(img,0.4) > 10
plt.imshow(_)
plt.show()
dmarks[fft_gradient(img,0.4) > 10] = 0
distance_transform_edt(dmarks)
plt.imshow(_)
plt.show()
d = _125
d[~T] = 0
plt.imshow(d)
plt.show()
d[~T] = 0
plt.imshow(d)
plt.show()
dt = d.copy()
from skimage.morphology import binary_dilation
dt = np.round(dt)
dt
dt.max()
dt.min()
BD = np.zeros(img.shape, np.int32)
#for r in range(1,14):
    #BD[dt==r] = binary_dilation(dt==r, rai]
from skimage.morphology import disk
for r in range(1,14):
    BD[dt==r] = r*binary_dilation(dt==r, selem=disk(int(r)))
    
for r in range(1,14):
    BD[dt==r] = r*binary_dilation(dt[dt==r], selem=disk(int(r)))
    
    
dt
dt.shape
dt==r
#for r in range(1,14):
#    BD[dt==r] = binary_dilation(dt==r, selem=disk(int(r))) 
    
BD = np.zeros((13,*img.shape), np.int32)
for r in range(14):
    BD[r] = binary_dilation(dt==r, selem=disk(int(r))) 
    
    
plt.imshow(B.max(axis=0))
plt.imshow(BD.max(axis=0))
plt.show()
dt==r
dt
plt.imshow(dt)
plt.show()
for r in range(13):
    BD[r] = binary_dilation(dt==(r+1), selem=disk(int(r+1)))
     
    
BD
BD.max()
dt==13 
    
plt.imshow(_)
plt.show()
dt==11

    
plt.imshow(_)
plt.show()
for r in range(13):
    BD[r] = binary_dilation(dt==(r+1), selem=disk(int(r+1)))
    
     
    
view_slices(BD)
plt.imshow(BD.max(axis=0))
plt.show()
from placenta import open_typefile
from placenta import open_tracefile
open_tracefile(filename)
trace = _
from scoring import confusion, mcc
confusion(BD.max(axis=0) > 0, trace)
plt.show()
plt.imshow(_)
plt.show()
