# coding: utf-8
plt.imshow(F.max(axis=-1))
plt.show()
plt.imshow(F.max(axis=-1)[crop])
plt.show()
fmg = F.max(axis=-1)[crop]
plt.imshow(fmg)
plt.show()
from scipy import ndimage as ndi
distance = ndi.distance_transoform_edt(fmg)
distance = ndi.distance_transform_edt(fmg)
distance
plt.imshow(distance)
plt.show()
from skimage.feature import peak_local_max
local_maxi = peak_local_max(distance, labels=fmg, footprint=np.ones((3,3)), indices=False)
local_maxi = peak_local_max(distance, labels=fmg,  indices=False)
local_maxi = peak_local_max(distance, indices=False)
local_maxi
plt.imshow(local_maxi)
plt.show()
local_maxi = peak_local_max(distance, footprint=np.ones((3,3)), indices=False)
plt.imshow(local_maxi)
plt.show()
markers = ndi.label(local_maxi)
markers
markers = ndi.label(local_maxi)[0]
from skimage.segmentation import watershed
watershed(-distance, markers, mask=fmg)
plt.imshow(_)
plt.show()
watershed(-distance, markers, mask=fmg)
watershed(-distance, markers)
plt.imshow(_)
plt.show()
from skimage.filters import sobel
sobel(fmg)
plt.imshow(_)
plt.show()
plt.imshow(watershed(gradient, markers=250, compactness=0.001))
plt.imshow(watershed(sobel(fmg), markers=250, compactness=0.001))
plt.show()
segs_w = watershed(sobel(fmg), markers=250, compactness=0.001)
from skimage.segmentation import mark_boundaries
plt.imshow(mark_boundaries(fmg,segs_w))
plt.show()
from skiamge.morphology import extrema
from skimage.morphology import extrema
extrema.local_maxima(fmg)
plt.imshow(_)
plt.show()
from skimage.measure import label
label_maxima = label(extrema.local_maxima(fmg))
plt.imshow(label_maxima)
plt.show()
from skimage.filters import rank
from morphology import disk
from skimage.morphology import disk
rank.gradient(fmg, disk(2))
plt.imshow(_)
plt.show()
markers = rank.gradient((255*fmg).astype('b'), disk(2)) < 10
plt.imshow(markers)
plt.show()
markers = rank.gradient((255*fmg), disk(2)) < 10
markers = rank.gradient(fmg, disk(2)) < 10
plt.imshow(markers)
plt.show()
markers = ndi.label(markers)[0]
markers
plt.imshow(_)
plt.show()
gradient = rank.gradient(fmg, disk(2))
labels = watershed(gradient, markers)
plt.imshow(labels)
plt.show()
plt.imshow(markers)
plt.show()
plt.imshow(gradient)
plt.show()
rank.median(fmg, disk(2))
plt.imshow(_)
plt.show()
rank.median(fmg, disk(1))
plt.show(_)
plt.show()
plt.imshow(_)
plt.show()
denoised = rank.median(fmg, disk(1))
from skimage.util import img_as_ubyte
get_ipython().magic('pinfo img_as_ubyte')
img_as_ubyte(fmg)
plt.show()
plt.imshow(_)
plt.show()
from skimage.exposure import rescale_intensity
get_ipython().magic('pinfo rescale_intensity')
rescale_intensity(fmg, [0,.2], [0,255])
rescale_intensity(fmg, in_range=[0,.2], out_range=[0,255])
rescale_intensity(fmg, in_range=np.array([0,.2]), out_range=np.array([0,255]))
help(rescale_intensity)
rescale_intensity(fmg, in_range=(0,.2), out_range=(0,255))
plt.imshow(_)
plt.show()
d = rescale_intensity(fmg, in_range=(0,.2), out_range=(0,255))
plt.imshow(d)
plt.show()
denoised = rank.median(d, disk(2))
denoised = rank.median(d.astype('uint'), disk(2))
plt.imshow(denoised)
plt.show()
markers = np.gradient(denoised, disk(5)) < 10
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]
plt.imshow(markers, camp=plt.cm.nipy_spectral, interpolation='nearest')
plt.imshow(markers, camp=plt.cm.nipy_spectral, interpolation='nearest')
plt.imshow(markers, camp=plt.cm.nipy_spectral)
plt.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
plt.show()
gradient = rank.gradient(denoised, disk(2))
plt.imshow(gradient)
plt.shw()
plt.show()
labels = watershed(gradient, markers)
plt.imshow(labels)
plt.show()
plt.imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
plt.show()
plt.imshow(gradient, cmap=plt.cm.nipy_spectral, interpolation='nearest')
plt.show()
from skimage.filters import threshold_local
threshold_local(fimg, 5, method='mean')
threshold_local(fmg, 5, method='mean')
plt.imshow(_)
plt.show()
plt.imshow(fimg)
plt.imshow(fmg)
plt.show()
threshold_local(fmg, 3, method='mean')
plt.imshow(_)
plt.show()
threshold_local(fmg, 3, method='median')
plt.imshow(_)
plt.show()
from skimage import filters
filters.threshold_minimum(fmg)
plt.imshow(fmg < _)
plt.show()
get_ipython().magic('pinfo filters.threshold_adaptive')
get_ipython().magic('pinfo filters.threshold_isodata')
filters.threshold_isodata(fmg)
fmg
fmg.max()
plt.imshow(fmg > .15356)
plt.show()
get_ipython().magic('pinfo filters.threshold_niblack')
filters.threshold_niblack(fmg, window_size=2)
filters.threshold_niblack(fmg, window_size=5)
plt.imshow(_)
plt.show()
get_ipython().magic('pinfo filters.thresholding')
get_ipython().magic('pinfo filters.threshold_yen')
filters.threshold_yen(fmg)
fmg > _
plt.imshow(_)
plt.show()
get_ipython().magic('pinfo filters.threshold_local')
filters.threshold_local(fmg)
filters.threshold_local(fmg,block_size=15)
plt.imshow(_)
plt.show()
plt.imshow(fmg)
plt.show()
sobel(fmg)
plt.imshow(_)
plt.show()
np.invert(sobel(fmg))
filters.invert(sobel(fmg))
get_ipython().magic('pinfo filters.inverse')
from skimage.morphology import h_maxima
h_maxima(fmg,1)
plt.imshow(_)
plt.show()
fmg
fmg.max()
rescale_intensity(fmg,out_range=np.uint8)
f = _
f.dtype
f.max()
plt.imshow(f)
plt.show()
h_maxima(f,100)
plt.imshow(_)
plt.show()
h_maxima(f,30)
plt.imshow(_)
plt.show()
plt.imshow(fmg > .5)
plt.show()
plt.imshow(fmg > .3)
plt.show()
markers = np.zeros(fmg.shape, dtype=np.uint8)
markers[data > .5] = 2
markers[fmg > .5] = 2
markers[fmg < .05] = 1
plt.imshow(markers)
plt.show()
from skimage.segmentation import random_walker
labels = random_walker(fimg, markers, beta=10, mode='bf')
labels = random_walker(fmg, markers, beta=10, mode='bf')
plt.imshow(labels)
plt.show()
markers[fmg > .9] = 2
markers[:] = 0
markers[fmg < 0.05] = 1
markers[fmg > .9] = 2
plt.imshow(markers)
plt.show()
plt.imshow(random_walker(fmg, markers))
plt.show()
plt.imshow(random_walker(fmg, markers, beta=10, mode='bf'))
plt.show()
plt.imshow(markers)
plt.show()
markers[fmg > .2] = 2
plt.imshow(random_walker(fmg, markers, beta=10, mode='bf'))
plt.show()
help(random_walker)
markers[:] = 0
plt.imshow(markers)
plt.show()
markers[fmg < .05]=-1
plt.imshow(markers)
plt.show()
markers[fmg < .03]=-1
markers[:] = 0
markers[fmg < .03]=-1
plt.imshow(markers)
markers[fmg > .25] = 2
markers[fmg <= .25] = 1
markers[fmg < .03] = -1
plt.imshow(markers)
plt.show()
markers
markers[:] = 0
plt.imshow(markers)
plt.show()
markers
maerkrs[fmg > .25] = 2
markers[fmg > .25] = 2
plt.imshow(markers)
plt.show()
markers
markers[fimg < .03]
markers[fmg < .03] = -1
markers
markers[:] = 0
markers[fmg < .2] = 1
markers[fmg < .05] = 0
markers[fmg > .5] = 2
plt.imshow(_)
plt.show()
markers[fmg < .2] = 0
markers[fmg > .2] = 1
markers[fmg > .5] = 2
plt.imshow(markers)
plt.show()
markers[:] += 1
markers[fmg < 0.03] = 0
plt.imshow(_)
plt.imshow(markers)
plt.show()
labels = random_walker(fmg, markers, beta=2, mode='bf')
plt.imshow(labels)
plt.show()
labels = random_walker(fmg, markers, beta=2, mode='bf')
labels = random_walker(fmg, markers, beta=3, mode='bf')
plt.imshow(labels)
plt.show()
labels = random_walker(fmg, markers, beta=.1, mode='bf')
plt.imshow(_)
plt.imshow(labels)
plt.show()
markers[:] = 0
markers[fmg < .01] = 1
markers[fmg > .25] = 2
plt.imshow(markers)
plt.show()
random_walker(fmg, markers, beta=2, mode='bf')
plt.imshow(_)
plt.show()
from score import compare_trace
get_ipython().magic('pinfo compare_trace')
approx = random_walker(fmg, markers, beta=2, mode='bf')
filename
compare_trace(approx, filename=filename)
open_typefile(filename, 'trace')
trace = _
compare_trace(approx,trace)
trace[crop]
trace = trace[crop]
compare_trace(approx,trace)
plt.imshow(_)
plt.show()
trace
plt.imshow(_)
plt.show()
plt.imshow(approx)
plt.show()
approx.shape
approx.dtype
approx
approx.astype('bool')
plt.imshow(_)
plt.show()
plt.imshow(approx)
plt.imshow()
plt.imshow(approx)
plt.show()
plt.imshow(approx != 0)
plt.show()
plt.imshow(approx > 0)
plt.show()
plt.imshow(approx !=255)
plt.show()
approx
plt.imshow(approx !=1)
plt.show()
plt.imshow(trace)
plt.show()
trace = np.invert(trace)
trace
plt.imshow(trace)
plt.show()
A = approx !=1
trace.max()
trace > 0
trace = _
A
plt.imshow(A)
plt.show()
plt.imshow(trace)
plt.show()
compare_trace(A,trace)
plt.show(_)
plt.show()
plt.imshow(_)
plt.show()
trace = np.invert(trace)
compare_trace(A,trace)
plt.show()
plt.imshow(_)
plt.show()
filename
compare_trace(A,trace)
plt.imshow(_)
plt.show()
plt.imshow(fmg)
plt.show()
ff = fmg.copy()
ff
ff[trace] = 0
plt.imshow(ff)
plt.show()
F2, _, _, _ = extract_pcsvn(filename, DARK_BG=True, alpha=.1, alphas=alphas, betas=betas, scales=scales, log_range=log_range, verbose=False, generate_graphs=False, generate_json=False, output_dir=OUTPUT_DIR, kernel='discrete')
F2.max(axis=-1)[crop]
plt.imshow(_)
plt.show()
fming = F2.max(axis=-1)[crop]
get_ipython().magic('pinfo np.choose')
get_ipython().magic('pinfo np.max')
get_ipython().magic('pinfo np.choose')
get_ipython().magic('pinfo np.max')
get_ipython().magic('pinfo np.fmax')
np.fmax(fmg,fming)
plt.imshow(_)
plt.show()
both = np.fmax(fmg,fming)
b = both.copy()
b[trace]=0
plt.imshow(b)
plt.show()
plt.imshow(both)
plt.show()
plt.imshow(img)
plt.show()
plt.imshow(img[crop])
plt.show()
sobel(img[crop])
plt.imshow(_)
plt.show()
plt.imshow(np.fmax(sobel(img)[crop],fmg))
plt.show()
sobel(img[crop]).max()
np.fmax(sobel(img)[crop],fmg)
edged = _
E = edged.copy()
E[trace] = 0
plt.imshow(E)
plt.show()
edged > .2
plt.imshow(_)
plt.show()
plt.imshow(edged > .1)
plt.show()
plt.imshow(edged > .15)
plt.show()
plt.imshow(edged)
plt.show()
plt.imshow(edged)
from skimage.morphology import reconstruction
reconstruction(fmg, edged)
R = _
plt.imshow(R)
plt.show()
plt.imshow(R>.15)
plt.show()
compare_trace(R>.15, A)
plt.imshow(_)
plt.show()
compare_trace(R>.15, trace)
plt.show()
plt.imshow(compare_trace(R>.15, trace))
plt.show()
reconstruction(fmg, edged, selem=disk(5))
R = _
plt.imshow(R)
plt.show()
plt.imshow(sobel(fmg))
plt.show()
plt.imshow(dobel(fmg))
plt.imshow(sobel(fmg))
plt.show()
