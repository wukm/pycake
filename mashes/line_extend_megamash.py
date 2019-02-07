# coding: utf-8
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', 'pycake/')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('run', 'margin_add_demo.py')
plt.show()
fig.show()
plt.imshow(approx)
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
plt.imshow(approx)
plt.imshow(confusion)
plt.imshow(confuse)
plt.imshow(confusion(approx,trace))
plt.imshow(confusion(approx,trace[crop]))
plt.imshow(confusion(approx,trace)[crop])
plt.imshow(confusion(approx,trace)[crop][200:400,600:800])
labs, FA
labs = F[:-4].argmax(axis=0)
plt.imshow(labs[crop])
plt.imshow(((f.max(axis=0) > .4)*labs)[crop])
plt.imshow(((f.max(axis=0) > .4)*labs)[crop])
plt.imshow(((f.max(axis=0) > 0)*labs)[crop])
plt.imshow(((f.max(axis=0) > .4)*labs)[crop])
plt.imshow(f.max(axis=0))
f.max()
plt.imshow(((F[:-4].max(axis=0) > .4)*labs)[crop])
mpl.rcParams['figure.figsize'] = (15,15)
plt.imshow(((F[:-4].max(axis=0) > .4)*labs)[crop])
labs = ((F[:-4].max(axis=0) > .4)*labs)
labs
plt.imshow(labs)
from importlib import reload
import diffgeo
reload(diffgeo)
labs.max()
labs.min()
scales
scales.shape
scales[:-4].shape
labs
labs+=1
labs*(f>.4)
labs*(f>.4)*img.mask
labs*(f>.4)*~img.mask
plt.imshow(_)
labs*(f>.4)*~dilate_boundary(None,mask=img.mask, radius=20)
plt.show()
plt.imshow(_)
plt.imshow(labs)
L = labs.copy()
L[dilate_boundary(None,mask=img.mask, radius=20)] = -1
L[f < .4] = -1
plt.imshow(L)
plt.imshow(L==-1)
plt.imshow(f < .4)
k=0; diffgeo.principal_directions(img, sigma=scales[k], mask=(L!=k))
T2, T1 = _
T2
plt.imshow(T2)
T2
T2.max()
T2.min()
labs==0
labs.any()
k=3; diffgeo.principal_directions(img, sigma=scales[k], mask=(L!=k))
T2, T1 = _
plt.imshow(T2)
plt.imshow(T2.filled(0))
k=3; diffgeo.principal_directions(img, sigma=scales[k], mask=(L==k))
k=3; diffgeo.principal_directions(img, sigma=scales[k], mask=(L!=k))
T2,  T1 = _
plt.imshow(T2)
plt.imshow(T1)
labs
plt.imshow(L)
plt.imshow(L==0)
plt.imshow(L==1)
plt.imshow(L==2)
plt.imshow(L==3)
plt.imshow(L==4)
plt.imshow(L==5)
plt.imshow(L==6)
plt.imshow(L==7)
plt.imshow(L==8)
plt.imshow(L==9)
plt.imshow(L==10)
plt.imshow(L==11)
plt.imshow(L==12)
plt.imshow(L==13)
plt.imshow(L==15)
thetas = [diffgeo.principal_directions(img, sigma=scales[k], mask=(L!=k))[0].filled(0) for k in range(len(scales))]
thetas
thetas = np.stack(thetas)
thetas.shape
T = thetas.max(axis=0)
plt.imshow(T)
plt.imshow(T, cmap='seismic')
plt.imshow(T, cmap='seismic')
plt.colorbar()
plt.imshow(T, cmap='seismic'); plt.colorbar()
plt.imshow(T, cmap='Spectral'); plt.colorbar()
degmap = plt.cm.Spectral
degmap.set_bad('k')
plt.imshow(ma.masked_array(T, L==-1), cmap='Spectral'); plt.colorbar()
plt.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); plt.colorbar()
mpl.rcParams['figure.figsize'] = (25,25)
plt.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); plt.colorbar(shrink=0.75)
fig, ax =  plt.subplots()
ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); plt.colorbar()
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); plt.colorbar(im)
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); plt.colorbar(im)
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im)
fig, ax =  plt.subplots()
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im)
fig.savefig('principal_directions_demo')
fig.show()
plt.show()
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.6)
fig.savefig('principal_directions_demo')
fig, ax =  plt.subplots()
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.6)
fig.savefig('principal_directions_demo')
fig, ax =  plt.subplots(figsize=(15,10))
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.6)
fig.savefig('principal_directions_demo')
fig, ax =  plt.subplots(figsize=(15,10))
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.6)
ax.axis('off')
fig, ax =  plt.subplots(figsize=(15,10))
im = ax.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.8)
ax.axis('off')
fig.savefig('principal_directions_demo')
#plt.imshow(ma.masked_array(T, L==-1)[crop], cmap='Spectral')
plt.imshow(nf > .05)
from skimage.filters import scharr
scharr(img)
plt.imshow(_)
scharr(img, mask=img.mask)
plt.imshow(_)
scharr(img, mask=~img.mask)
plt.show()
plt.imshow(_)
plt.imshow(scharr(img, mask=~img.mask)[crop], cmap='gray')
s = scharr(img, mask=~img.mask)
plt.imshow((s > .005)[crop], cmap='gray')
plt.imshow((s > .01)[crop], cmap='gray')
plt.imshow((s < .02)[crop], cmap='gray')
plt.imshow((s < .05)[crop], cmap='gray')
plt.imshow((s < .01)[crop], cmap='gray')
plt.imshow((s < .03)[crop], cmap='gray')
T
plt.imshow(T)
T[L==-1] = -1
plt.imshow(T)
thin(T!=-1)
plt.imshow(_)
tt = thin(T!=-1)*T
plt.imshow(_)
tt = thin(T!=-1)*T
plt.imshow(tt)
tt = thin(T!=-1)*T
tt[T==-1] = -1
plt.imshow(tt)
tt[thin(T!=-1)] = -1
tt
plt.imshow(tt)
tt = thin(T!=-1)*T
tt[~thin(T!=-1)] = -1
pl.imshow(tt)
plt.imshow(tt)
fig, ax =  plt.subplots(figsize=(15,10))
im = ax.imshow(ma.masked_array(tt, tt==-1)[crop], cmap='Spectral'); fig.colorbar(im, shrink=0.8)
fig.savefig('principal_directions_thinned_demo')
plt.imshow('pd_thinned.png' ,ma.masked_array(tt, tt==-1)[crop], cmap='Spectral')
plt.imsave('pd_thinned.png', ma.masked_array(tt, tt==-1)[crop], cmap='Spectral')
plt.imshow('pd.png' ,ma.masked_array(T, L==-1)[crop], cmap='Spectral')
plt.imsave('pd.png' ,ma.masked_array(T, L==-1)[crop], cmap='Spectral')
tt
tt > -1
from skimage.filters.rank import pop as local_pop
np.ones(3)
square = np.ones((3,3),dtype=np.bool)
square
local_pop(tt>-1, selem=square)
tt
tt>-1
tt!=-1
help(local_po)
help(local_pop)
from skimage.filters.rank import sum as local_count
local_count(tt>-1, selem=square)
plt.imshow(_)
lc = local_count(tt>-1, selem=square)
plt.imshow((lc > 0) & (lc < 2))
plt.imshow(tt>-1)
plt.imshow((lc > 0) & (lc < 3))
plt.imshow((lc > 0) & (lc < 2))
plt.imshow((lc < 2))
plt.imshow((lc > 2))
plt.imshow((lc < 3))
plt.imshow((lc < 3) & (lc >=2))
plt.imshow((lc ==2))
plt.imshow((lc==1))
plt.imshow((lc==3))
plt.imshow((lc==4))
plt.imshow((lc==5))
plt.imshow((lc==1))
plt.imshow((lc==3))
plt.imshow(lc==4)
plt.imshow(lc==5)
plt.imshow(lc==9)
plt.imshow(lc==8)
plt.imshow(lc==7)
plt.imshow(lc==6)
plt.imshow(lc==5)
plt.imshow(lc==4)
plt.imshow(lc)
plt.imshow(lc==0)
lc.max()
lc = local_count((tt>-1).astype('bool'), selem=square)
lc.max()
(tt>-1).astype('uint8')
_.max()
lc = local_count((tt>-1).astype('uint8'), selem=square)
lc.max()
plt.imshow((lc>1) & (lc < 3))
plt.imshow(lc==2)
plt.imshow((lc==2)&(tt>-1))
plt.imshow(((lc==2)&(tt>-1))[crop])
plt.imshow(((lc==1)&(tt>-1))[crop])
endpoints = ((lc==1)&(tt>-1))
plt.imshow(endpoints)
plt.imshow(np.maximum(5*endpoints, tt))
plt.imsave('pd_endpoints.png',((lc==1)&(tt>-1))[crop])
plt.imsave('pd_endpoints.png',((lc==2)&(tt>-1))[crop])
endpoints = ((lc==2) & (tt >-1))
plt.imshow(endpoints)
tt
plt.imshow(tt)
endpoint_thetas = tt.copy()
endpoint_thetas[~endpoints] = -1
plt.imshow(endpoints)
plt.imshow(endpoint_thetas)
L*endpoints
plt.imshow(_)
mask
from functools import partial
mask = lambda a: ma.masked_array(a, a!=-1)
plt.imshow(mask(endpoint_thetas), 'Spectral')
mask = lambda a: ma.masked_array(a, a==-1)
plt.imshow(mask(endpoint_thetas), 'Spectral')
plt.imshow(mask(endpoint_thetas)[crop], 'Spectral')
plt.imshow(mask(endpoint_thetas)[crop], 'Spectral')
plt.imshow(mask(endpoint_thetas)[crop][200:400][600:800], 'Spectral')
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], 'Spectral')
tt
plt.imshow(_)
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], 'Spectral'); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=Spectral_r); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.Spectral_r); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.twilight); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.twilight_shifted_r); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.hsv); plt.colorbar()
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.hsv); plt.colorbar()
hsv = plt.cm.hsv
hsv.set_bad('k')
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=hsv); plt.colorbar()
plt.imsave('pd_endpoints.png',((lc==2)&(tt>-1))[crop], plt.cm.gray)
plt.imsave('pd_endpoints.png',((lc==2)&(tt>-1))[crop], cmap=plt.cm.gray)
get_ipython().run_line_magic('hist', '-g plt.imsave')
get_ipython().run_line_magic('175', '')
plt.imsave('pd.png' ,ma.masked_array(T, L==-1)[crop], cmap=hsv)
plt.imsave('pd_thinned.png', ma.masked_array(tt, tt==-1)[crop], cmap=hsv)
plt.imsave('pd_endpoints.png', ma.masked_array(endpoint_thetas, endpoint_thetas==-1)[crop], cmap=hsv)
plane = np.zeros((100,100))
from skimage.draw import line
np.sin(30)
plane[line(50,30, 50 - 5*np.sin(np.pi/3), 30 - 5*np.cos(np.pi/3))] = 1
plane[line(50,30, int(50 - 5*np.sin(np.pi/3)), int(30 - 5*np.cos(np.pi/3)))] = 1
plane
plt.imshow(plane)
plane[line(50,30, int(5*np.sin(np.pi/3)-50), int(5*np.cos(np.pi/3))-30)] = 1
plane = np.zeros((100,100))
plane[line(50,30, int(5*np.sin(np.pi/3)-50), int(5*np.cos(np.pi/3))-30)] = 1
plt.imshow(plane)
plane = np.zeros((100,100))
plane[line(50,30, int(5*np.cos(np.pi/3)-50), int(5*np.sin(np.pi/3))-30)] = 1
plt.imshow(plane)
plane = np.zeros((100,100))
plane[line(50,30, int(5*np.cos(np.pi/2)-50), int(5*np.sin(np.pi/2))-30)] = 1
plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30, 50 - int(5*np.sin(np.pi/2)), 30 - int(5*np.cos(np.pi/2)))] = 1
plane = np.zeros((100,100)); plane[line(50,30, 50 - int(5*np.sin(np.pi/2)), 30 - int(5*np.cos(np.pi/2)))] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30, 50 - int(5*np.sin(np.pi/3)), 30 - int(5*np.cos(np.pi/3)))] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30, 50 + int(5*np.sin(np.pi/3)), 30 + int(5*np.cos(np.pi/3)))] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,50 + int(5*np.sin(np.pi/3)), 30 + int(5*np.cos(np.pi/3)))] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi/3))-50,  int(5*np.cos(np.pi/3))-30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi/3))-30,  int(5*np.cos(np.pi/3))-50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi/3))+30,  int(5*np.cos(np.pi/3))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi/2))+30,  int(5*np.cos(np.pi/2))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi/2))+30,  int(5*np.cos(np.pi/2))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(5*np.sin(np.pi))+30,  int(5*np.cos(np.pi))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi))+30,  int(2*np.cos(np.pi))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi))+30,  int(2*np.cos(np.pi))+50)] = 1; plt.imshow(plane)
help(np.meshgrid)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi))+30,  int(2*np.cos(np.pi))+50)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi))+50,  int(2*np.cos(np.pi))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi/2))+50,  int(2*np.cos(np.pi/2))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(2*np.sin(np.pi/3))+50,  int(2*np.cos(np.pi/3))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(np.pi/3))+50,  int(10*np.cos(np.pi/3))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(0))+50,  int(10*np.cos(0))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(np.pi/6))+50,  int(10*np.cos(np.pi/6))+30)] = 1; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(0))+50,  int(10*np.cos(0))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(0))+50,  int(10*np.cos(0))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(10*np.sin(np.pi/2))+50,  int(10*np.cos(np.pi/2))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(np.pi/2))+50,  int(-10*np.cos(np.pi/2))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(-10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))-50,  int(-10*np.cos(theta))-30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(-10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(-10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(-10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(-10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))-50,  int(-10*np.sin(theta))-30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))-50,  int(10*np.sin(theta))-30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))-50,  int(10*np.sin(theta))-30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))+50,  int(10*np.sin(theta))-30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))-50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(10*np.cos(theta))+50,  int(-10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/2; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/2; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = 0; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.cos(theta))+50,  int(10*np.sin(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = 0; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/2; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = 2*np.pi/3; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = 3*np.pi/4; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
theta = np.pi; plane = np.zeros((100,100)); plane[line(50,30,int(-10*np.sin(theta))+50,  int(10*np.cos(theta))+30)] = 1; plane[50,30]=2; plt.imshow(plane)
def line_coords(r0,c0, theta, l=10): return line(r0,c0,int(-l*np.sin(theta))+r0,  int(l*np.cos(theta))+c0)
endpoint_thetas
plt.imshow(mask(endpoint_thetas)[crop][200:400,600:800], cmap=plt.cm.hsv); plt.colorbar()
N = endpoint_thetas.copy()
np.where(endpoint_thetas != -1)
for r,c in np.where(endpoint_thetas != -1):
    print(f'({r},{c})', end=' ')
for r,c in *np.where(endpoint_thetas != -1):
    print(f'({r},{c})', end=' ')
for r,c in zip(np.where(endpoint_thetas != -1)):
    print(f'({r},{c})', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    print(f'({r},{c})', end=' ')
for r,c in zip(np.where(endpoint_thetas != -1)):
    print(f'({r},{c}): endpoint_thetas[r,c]', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    print(f'({r},{c}): endpoint_thetas[r,c]', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    print(f'({r},{c}): {endpoint_thetas[r,c]}', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    print(f'({r},{c}): {endpoint_thetas[r,c]:.2f}', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    print(f'({r},{c}): {endpoint_thetas[r,c]:.2f}', end=' ')
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c],l=2)] = 0
plt.imshow(N)
plt.imshow(N, cmap=hsv)
N
N.max()
plt.imshow(N)
N = endpoint_thetas.copy()
plt.imshow(N, cmap=hsv)
hsv.set_bad('k')
plt.imshow(mask(N), cmap=hsv)
plt.imshow(mask(N)[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c],l=2)] = 0
plt.imshow(mask(N)[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c],l=5)] = 0
plt.imshow(mask(N)[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c],l=5)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, np.pi/2 +  endpoint_thetas[r,c],l=5)] = 0
plt.imshow(mask(N)[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, np.pi/2 +  endpoint_thetas[r,c],l=10)] = 0
plt.imshow(_)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, np.pi/2 +  endpoint_thetas[r,c],l=10)] = 0
plt.imshow(mask(N)[crop], cmap=hsv)
plt.imshow(mask(np.maximum(N,endpoint_thetas))[crop], cmap=hsv)
plt.imsave('pd_extend.png', mask(np.maximum(N,endpoint_thetas))[crop], cmap=hsv)
plt.imshow(mask(np.maximum(N,tt))[crop], cmap=hsv)
plt.imsave('pd_extend_all.png', mask(np.maximum(N,endpoint_thetas))[crop], cmap=hsv)
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, -np.pi/2 +  endpoint_thetas[r,c],l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, np.pi/2 -  endpoint_thetas[r,c],l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c] - np.pi/2,l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c],l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c] + np.pi/2,l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c] + np.pi/2,l=10)] = 0
plt.imsave('pd_extend_all.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c] - np.pi/2,l=10)] = 0
plt.imsave('pd_extend_alt.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, endpoint_thetas[r,c] - np.pi/4,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, 2*(endpoint_thetas[r,c] - np.pi/2),l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    N[line_coords(r,c, np.abs(endpoint_thetas[r,c] - np.pi/2),l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    if theta > np.pi:
        theta = np.pi - theta 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    if theta > np.pi:
        theta = np.pi/2 - theta 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    if theta > np.pi:
        theta = np.pi - theta + np.pi/2 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    if theta > np.pi:
        theta = np.pi - theta 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al4.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/2
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al3.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
endpoint_thetas.min()
endpoint_thetas[endpoint_thetas != -1].min()
ni.pi/2
np.pi/2
np.pi/4
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] - np.pi/4
    N[line_coords(r,c, theta ,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] - np.pi/4
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/4
    
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + 3*np.pi/4
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta ,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + 3*np.pi/4
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] - 3*np.pi/4
    N[line_coords(r,c, theta ,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] - np.pi/2
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/4
    if theta > np.pi:
        theta -= np.pi 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + np.pi/4
    if theta > np.pi:
        theta -= np.pi 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c] + 3*np.pi/4
    if theta > np.pi:
        theta -= np.pi 
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
get_ipython().run_line_magic('hist', '')
thetas = [diffgeo.principal_directions(img, sigma=scales[k], mask=(L!=k))[0].filled(0) for k in range(len(scales))]
thetas = np.stack(thetas)
thetas
L
L[L!=-1]
L*(L!=-1)
thetas[L*(L!=-1)]
thetas[L*(L!=-1),...]
get_ipython().run_line_magic('pinfo', 'np.choose')
labs
labs
thetas[labs]
help(np.select)
help(np.compress)
np.compress(labs, thetas, axis=0)
help(np.compress)
help(np.extract)
np.take(thetas, labs, axis=0)
thetas
thetas.shape
labs
labs.data
np.take(thetas, labs.data, axis=0)
np.take_along_axis(thetas, labs, 0)
np.take_along_axis(thetas, np.atleast_3d(labs), 0)
np.take_along_axis(thetas, labs[None,...], 0)
_.shape
np.take_along_axis(thetas, labs[None,...], 0)
the = _
the.max()
the.min()
the.squeeze()
T = _
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = endpoint_thetas[r,c]
    N[line_coords(r,c, theta ,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    N[line_coords(r,c, theta ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    N[line_coords(r,c, theta - np.pi/2 ,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    N[line_coords(r,c, theta - np.pi/2 ,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c]
    if theta > np.pi:
        theta -= np.pi / 2
    N[line_coords(r,c, theta,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
T.min()
T.max()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c] + np.pi/2
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta,l=10)] = 0
N = endpoint_thetas.copy()
for r,c in zip(*np.where(endpoint_thetas != -1)):
    theta = T[r,c] + np.pi/2
    if theta > np.pi:
        theta -= np.pi
    N[line_coords(r,c, theta,l=10)] = 0
plt.imsave('pd_extend_al2.png', mask(np.maximum(N,tt))[crop], cmap=hsv)
