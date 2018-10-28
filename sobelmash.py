# coding: utf-8
get_ipython().magic('ls ')
get_ipython().magic('who ')
f
F
F.max(axis=-1)
Fmax = _
plt.imshow(Fmax)
plt.show()
plt.imshow(Fmax[crop], cmap=plt.cm.spec)
plt.imshow(Fmax[crop], cmap=plt.cm.spectral)
plt.show()
fmax = Fmax.copy()
trace
trace!=0
trace==0
fmax(trace==0)
fmax[trace==0]
fmax[trace==0] = 0
plt.imshow(fmax)
plt.show()
plt.imshow(fmax[crop])
plt.show()
get_ipython().magic('pinfo confusion')
get_ipython().magic('pinfo confusion')
confusion_4()
from score import confusion_4
get_ipython().magic('pinfo confusion_4')
confusion_4(approx, truth)
confusion_4(approx, trace)
plt.imshow(_)
plt.show()
plt.imshow(img)
plt.show()
from skimage.filters import sobel
get_ipython().magic('pinfo sobel')
sobel(img)
plt.imshow(_)
plt.show()
plt.imshow(sobel(img)[crop])
plt.show()
f_max
fmax
Fmax
plt.imshow(Fmax)
plt.show()
get_ipython().magic('pinfo np.expand_dims')
Fmax[:,:,np.newaxis]
np.vstack((Fmax,Fmax,Fmax))
_.shape
np.dstack((Fmax,Fmax,Fmax)).shape
np.dstack((sobel(img),Fmax,Fmax))
comb = _
plt.imshow(comb)
plt.show()
plt.imshow(comb[crop])
plt.show()
comb = np.dstack((sobel(img),np.zeros_like(Fmax),Fmax))
plt.imshow(comb[crop])
plt.show()
sobel(img).max()
S = sobel(img)
S /= S.max()
S
plt.imshow(S)
plt.show()
from plate_morphology import dilate_boundary
get_ipython().magic('pinfo dilate_boundary')
dilate_boundary(S, mask=img.mask)
plt.imshow(_)
plt.show()
dilate_boundary(S, mask=img.mask).filled(0)
plt.imshow(_[crop])
plt.show()
S = dilate_boundary(S, mask=img.mask).filled(0)
plt.imshow(_[crop])
plt.imshow(S[crop])
plt.show()
comb
comb = np.dstack((S,np.zeros_like(Fmax),Fmax))
plt.show()
plt.imshow(comb)
plt.show()
comb = np.dstack((Fmax / Fmax.max(),S,np.zeros_like(Fmax)))
plt.imshow(comb[crop])
plt.show()
comb = np.dstack((Fmax / Fmax.max(),S, Fmax))
plt.imshow(comb[crop])
plt.show()
comb.max()
comb.min()
s
S
plt.imshow(S, vmin=0, vmax=1)
plt.show()
Z = np.zeros_like(S)
plt.imshow(np.dstack(Z,S,Z))
plt.imshow(np.dstack((Z,S,Z)))
plt.show()
plt.imshow(np.dstack((Z,S*2,Z)))
plt.show()
plt.imshow(np.dstack((Z,S*2,Fmax)))
plt.show()
get_ipython().magic('save sobelmash 1-103')
