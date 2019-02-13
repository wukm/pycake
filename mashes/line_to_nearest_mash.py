# coding: utf-8
get_ipython().run_line_magic('run', 'basic.py')
F = [frangi_from_image(img, sigma, beta=0.15, gamma=0.5, dark_bg=True, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5, 3.5, num=20, base=2)]
F = np.stack(F)
F
Fmax = F[:-4].max(axis=0)
plt.imshow(Fmax)
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (25,25)
mpl.rcParams['figure.figsize'] = (20,20)
plt.imshow(Fmax)
F = np.stack([frangi_from_image(img, sigma, beta=0.15, gamma=0.5, dark_bg=False, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5, 3.5, num=20, base=2)])
Fmax = F[:-4].max(axis=0)
plt.imshow(Fmax)
imshow = plt.imshow
imshow(Fmax[crop])
from scipy.ndimage import watershed_ift
markers = np.zeros(img.shape, np.uint8)
markers[Fmax > .4] = 1
markers[Fmax == 0] = -1
watershed_ift(Fmax, markers)
Fmax
from scipy.util import img_as_uint8
from skimage.util import img_as_uint
img_as_uint(Fmax)
watershed_ift(img_as_uint(Fmax), markers)
watershed_ift(img_as_uint(Fmax), img_as_int(markers))
from skimage.util import img_as_int
watershed_ift(img_as_uint(Fmax), img_as_int(markers))
plt.imshow(_)
markers
markers[Fmax == 0] = 2
watershed_ift(img_as_uint(Fmax), img_as_int(markers))
plt.imshow(_)
watershed_ift(img_as_uint(Fmax)[crop], img_as_int(markers))
imshow(watershed_ift(img_as_uint(Fmax), img_as_int(markers))[crop])
W = watershed_ift(img_as_uint(Fmax), img_as_int(markers))
W
W!=257
plt.imshow(_)
W[Fmax>.4]
W==128
plt.imshow(_)
imshow(np.maximum(1.*(W==128), 2*(Fmax > .4)))
imshow(np.maximum(1.*(W==128), 2.*(Fmax > .4)))
np.maximum(1.*(W==128), 2.*(Fmax > .4))
_.data.dtype
np.maximum(1.*(W==128), 2.*(Fmax > .4)).max()
np.maximum(1.*(W==128), 2.*(Fmax > .4)) == 2
_.any()
np.maximum(1.*(W==128), 2.*(Fmax > .4)) == 2
np.maximum(1.*(W==128), 2.*(Fmax > .4)) == 1
_.any()
np.maximum(1.*(W==128), 2.*(Fmax > .4)) == 1
plt.imshow(_)
W = watershed_ift(img_as_uint(1-Fmax), img_as_int(markers))
plt.imshow(W)
Fmax
from skimage.segmentation import thin
from skimage.morphology import thin
thin(Fmax)
tf =  _
plt.imshow(tf[crop])
tf =  thin(Fmax > .4)
plt.imshow(tf)
from skimage.morphology import square
from scipy.ndimage import sum as local_count
from skimage.filters.rank import sum as local_count
local_count(tf, square(3))
plt.imshow(_)
local_count(tf, square(3))
_.max()
tf.max()
local_count(tf.astype('int'), square(3))
_.sum()
local_count(tf.astype('int'), square(3))
_.max()
local_count(tf.astype('uint'), square(3))
neighbors = _
neighbors==2
plt.imshow(_)
neighbors==1
plt.imshow(_)
(neighbors==1) & tf
plt.imshow(_)
endpoints = ((neighbors==1) & tf)
from scipy.ndimage import distance_transform_transform_edt as edt
from scipy.ndimage import distance_transform_edt as edt
edt(~endpoints)
plt.imshow(_)
edt(~endpoints) & endpoints
edt(~endpoints)*endpoints
plt.imshow(_)
edt(~endpoints)*endpoints
_.any()
for point in zip(*np.where(endpoints)):
    print(point)
zip(*np.where(endpoints))
list(zip(*np.where(endpoints)))
endlist = list(zip(*np.where(endpoints)))
for point in endlist:
    print(point, end=': ')
    print(min(((point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist)))
for point in endlist:
    print(point, end=': ')
    print(min(((point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point)))
get_ipython().run_line_magic('pinfo', 'min')
for point in endlist:
    print(point, end=': ')
    print(np.argmin(((point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point)))
for point in endlist:
    print(point, end=': ')
    print(np.argmin(np.array([point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point]))
for point in endlist:
    print(point, end=': ')
    print(np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point])))
for point in endlist:
    print(point, end=': ')
    endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point]))]
endlist
len(_)
for point in endlist:
    print(point, end=': ')
    endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 for p in endlist if p!=point else 10000]))]
for point in endlist:
    print(point, end=': ')
    endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
for point in endlist:
    print(point, end=': ')
    print(endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))])
from skimage.draw import line
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    
lines = np.zeros(*img.shape, np.uint8)
lines = np.zeros(img.shape, np.uint8)
lines
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    lines[line(*point, *nearest)] = 1
plt.imshow(lines)
imshow(np.maximum(2.*tf, 1.*lines))
imshow(np.maximum(2.*tf, 1.*lines)[crop])
endpoints
plt.imshow(endpoints)
plt.imshow(np.maximum(2.*(neighbors==1),1.*tf))
endpoints = ((neighbors==2) & tf)
plt.imshow(endpoints)
endlist = list(zip(*np.where(endpoints)))
lines = np.zeros(img.shape, np.uint8)
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    lines[line(*point, *nearest)] = 1
plt.imshow(lines)
plt.imshow(np.maximum(lines*1., tf*2.))
plt.imshow(np.maximum(lines*1., tf*2.)[crop])
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    lines[line(*point, *nearest)] = 1
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    to_add = line(*point, *nearest)
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    line_to_add = line(*point, *nearest)
    print(Fmax[line_to_add])
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    line_to_add = line(*point, *nearest)
    print(Fmax[line_to_add].mean())
lines = np.zeros(img.shape, np.uint8)
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    line_to_add = line(*point, *nearest)
    lines[line_to_add] = int(10*Fmax[line_to_add].mean())
plt.imshow(lines)
plt.imshow(np.maximum(lines,10*tf)); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines,10*tf)[crop]); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines,10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum((lines>=2)*lines,10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines,10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines(lines>=2),10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines*(lines>=2),10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5)
plt.imshow(np.maximum(lines*(lines>=2),10*tf)[crop],plt.cm.nipy_spectral); plt.colorbar(shrink=0.5); plt.savefig('nearest_endpoints_with_Fmaxmean')
connected = np.maximum(lines*(lines>=2),10*tf)
from scoring import confusion
from placenta import open_tracefile
trace = open_tracefile(filename)
plt.imshow(confusion(connected!=0, trace)[crop])
lines = np.zeros(img.shape, np.uint8)
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    line_to_add = line(*point, *nearest)
    lines[line_to_add] = int(10*Fmax[line_to_add].mean())
endpoints = ((neighbors>0) & (neighbors<3) & tf)
plt.imshow(endpoints)
endlist = list(zip(*np.where(endpoints)))
lines = np.zeros(img.shape, np.uint8)
for point in endlist:
    nearest = endlist[np.argmin(np.array([(point[0]-p[0])**2 + (point[1]-p[1])**2 if p!=point else 10000 for p in endlist]))]
    line_to_add = line(*point, *nearest)
    lines[line_to_add] = int(10*Fmax[line_to_add].mean())
plt.imshow(lines)
connected = np.maximum(lines*(lines>=2),10*tf)
plt.imshow(connected)
plt.imshow(connected, plt.cm.nipy_spectral)
plt.imshow(connected[crop], plt.cm.nipy_spectral)
plt.imshow(confusion(connected!=0, trace))
plt.imshow(confusion(connected!=0, trace)[crop])
from network_completion import mean_colored_connections
import network_completion
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
from importlib import reload
reload(network_completion)
mean_colored_connections(endpoints, Fmax)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
reload(network_completion)
mean_colored_connections(endpoints, Fmax)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
plt.imshow(_)
_.shape
endpoints
plt.imshow(_)
mean_colored_connections(endpoints, Fmax)
reload(network_completion)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
reload(network_completion)
reload(network_completion)
from network_completion import mean_colored_connections
mean_colored_connections(endpoints, Fmax)
plt.imshow(_)
connected = mean_colored_connections(endpoints, Fmax)
connected.min()
connected.max()
new_skel = np.maximum(connected,tf)
plt.imshow(new_skel)
plt.imshow(new_skel[crop], cmap=nipy_spectral)
plt.imshow(new_skel[crop], cmap='nipy_spectral')
plt.imshow(new_skel[crop], cmap='nipy_spectral'); plt.colorbar(shrink=0.5)
plt.imshow(new_skel[crop], cmap='nipy_spectral'); plt.colorbar(shrink=0.5)
from network_completion import find_endpoints
find_endpoints(new_skel >= .2)
reload(network_completion)
from network_completion import find_endpoints
find_endpoints(new_skel >= .2)
plt.imshow(_)
plt.imshow(new_skel > .2)
reload(network_completion)
from network_completion import find_endpoints
find_endpoints(new_skel >= .2)
plt.imshow(_)
new_endpoints = find_endpoints(new_skel >= .2)
mean_colored_connections(new_endpoints, Fmax)
new_connections = _
plt.imshow(new_connections)
plt.imshow(np.maximum(new_connections, new_skel))
get_ipython().run_line_magic('pinfo', 'np.clip')
help(np.ufuncs)
help(np.clip)
help(np.doc.ufunc)
help(np.doc.ufuncs)
np.info(np.clip)
get_ipython().run_line_magic('save', 'line_to_nearest_mash.py 1-227')
tf
neighbors = local_count(tf.astype('uint8'), np.ones((3,3),np.bool))
np.isin(neighbors,[1,2])
plt.imshow(_)
tf & (np.isin(neighbors,[1,2]))
plt.imshow(_)
tf*neighbors
plt.imshow(_)
tf&(neighbors==1)
np.argmin()
np.argsort()
get_ipython().run_line_magic('pinfo', 'np.argsort')
help(np.lexsort)
help(np.argpartition)
np.argpartition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first, second, _ = np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first, second, *_ = np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first
second
_
first, second, ... = np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first, second, *... = np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first, second, *... = np.partition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first, second, *__ = np.argpartition([0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.4], 2)
first
second
reload(network_completion)
from network_completion import mean_colored_connections
mean_colored_components(tf)
mean_colored_connections(tf)
reload(network_completion)
from network_completion import mean_colored_connections
mean_colored_connections(tf)
reload(network_completion)
from network_completion import mean_colored_connections
mean_colored_connections(tf)
plt.imshow(_)
lines = mean_colored_connections(tf)
plt.imshow(lines)
neighbors
reload(network_completion)
from network_completion import mean_colored_connections
lines = mean_colored_connections(tf)
plt.imshow(lines)
lines = mean_colored_connections(tf, Fmax)
plt.imshow(_)
plt.imshow(lines)
plt.imshow(np.maximum(tf, lines))
plt.imshow(np.maximum(tf, lines)[crop])
plt.imshow(np.maximum(tf, lines)[crop], plt.cm.nipy_spectral)
plt.imshow(np.maximum(tf, lines)[crop], plt.cm.nipy_spectral)
plt.imshow((lines>=.3)|tf)
confusion(_, trace)
del _
completed = (lines>=.3)|tf
confusion(test, approx)
confusion(completed, approx)
confusion(completed, trace)
plt.imshow(_)
plt.imshow(confusion(completed, trace)[crop])
morelines = mean_colored_connections(completed)
plt.imshow(morelines)
morelines = mean_colored_connections(completed, Fmax)
plt.imshow(morelines)
plt.imshow(np.maximum(morelines,completed), cmap=plt.cm.nipy_spectral)
plt.imshow(np.maximum(morelines>.3,completed), cmap=plt.cm.nipy_spectral)
plt.imshow(np.maximum(morelines>.2,completed), cmap=plt.cm.nipy_spectral)
confusion(np.maximum(morelines>.2,completed))
plt.show()
confusion(np.maximum(morelines>.2,completed))
plt.imshow(_298)
confusion(np.maximum(morelines>.2,completed), trace)
plt.imshow(_300[crop])
from postprocessing import dilate_to_rim
get_ipython().run_line_magic('pinfo', 'dilate_to_rim')
Fn = np.stack([frangi_from_image(img, sigma, beta=0.15, gamma=0.5, dark_bg=True, dilation_radius=20, rescale_frangi=True) for sigma in np.logspace(-1.5, 3.5, num=20, base=2)])
plt.imshow(Fn.max(axis=0))
plt.imshow(Fn.max(axis=0)>.05)
fmarg = Fn.max(axis=0) > .05
more_completed = np.maximum(morelines>.2,completed)
dilate_to_rim(more_completed, fmarg)
plt.imshow(_)
plt.imshow(_309)
dilate_to_rim(more_completed, fmarg)
D = _312
plt.imshow(confusion(D, approx))
plt.imshow(confusion(D, trace))
plt.imshow(confusion(D, trace)[crop])
plt.imshow(Fmax)
plt.imshow(Fmax[crop])
plt.imshow(np.maximum(Fmax,more_completed)[crop])
plt.imshow(np.maximum(Fmax,D)[crop])
plt.imshow(np.maximum(Fmax,more_completed)[crop])
plt.imsave('completed_by_nearest_to_line_mash', np.maximum(Fmax,more_completed)[crop])
get_ipython().run_line_magic('save', 'line_to_nearest_mash.py 1-322')
reload(network_completion)
from network_completion import colored_connections_max_path
tf
plt.imshow(tf)
colored_connections_max_path(tf, Fmax)
manylines = _
manylines = _328
plt.imshow(np.maximum(manylines, tf))
plt.imshow(np.maximum(manylines, tf)[crop])
plt.imshow(np.maximum(manylines>.2, tf)[crop])
plt.imshow(np.maximum(manylines>.3, tf)[crop])
plt.imshow(np.maximum(manylines>.4, tf)[crop])
plt.imshow(np.maximum(manylines>.3, tf)[crop])
plt.imshow(np.maximum(manylines>.3, tf)[crop])
plt.imshow(thin(np.maximum(manylines>.3, tf))[crop])
plt.imshow(np.maximum(manylines>.3, tf)[crop])
help(np.argpartition)
help(np.argmax)
reload(network_completion)
reload(network_completion)
reload(network_completion)
reload(network_completion)
from network_completion import colored_connections_max_path, colored_connections_any_nonzero
colored_connections_max_path(tr, scores)
colored_connections_max_path(tf, Fmax)
maxpathlines = _
maxpathlines = _348
plt.imshow(maxpathlines)
plt.imshow(maxpathlines[crop])
reload(network_completion)
from network_completion import colored_connections_max_path, colored_connections_any_nonzero
maxpathlines = colored_connections_max_path(tf, Fmax)
plt.imshow(maxpathlines)
from network_completion import colored_connections_max_path, colored_connections_any_nonzero
reload(network_completion)
from network_completion import colored_connections_max_path, colored_connections_any_nonzero
maxpathlines = colored_connections_max_path(tf, Fmax)
reload(network_completion)
from network_completion import colored_connections_max_path, colored_connections_any_nonzero
maxpathlines = colored_connections_max_path(tf, Fmax)
plt.imshow(maxpathlines)
maxpathlines = colored_connections_max_path(tf, F[:-4].max(axis=0))
imshow(maxpathlines[crop])
imshow(confusion(maxpathlines!=0, trace)[crop])
imshow(confusion(maxpathlines>.3, trace)[crop])
plt.imshow(tf)
imshow(confusion(tf | (maxpathlines>.3), trace)[crop])
connected = tf | (maxpathlines>.3)
plt.imshow(connected)
dilate_to_rim(connected, fmarg)
D = _
D = _373
plt.imshow(_)
plt.imshow(D)
plt.imshow(confusion(D,trace))
plt.imshow(confusion(D,trace))
from scoring import mcc
mcc(D, trace, bg_mask=img.mask)
thin(D)
plt.imshow(thin(D))
plt.imshow(maximum(2*thin(D),tf))
plt.imshow(np.maximum(2*thin(D),tf))
plt.imshow(np.maximum(2.*thin(D),1.*tf))
plt.imshow(np.maximum(2.*thin(D),1.*tf)[200:600,600:1000])
mean_colored_connections(tf, Fmax, double_connect=False)
plt.imshow(_388)
mean_colored_connections(tf, Fmax, double_connect=False)
for i in range(5):
    meanlines = mean_colored_connections(meanlines, Fmax, double_connect=False)
    meanlines = meanlines > .3
    
new_skel = tf.copy()
for i in range(5):
    meanlines = mean_colored_connections(new_skel, Fmax, double_connect=False)
    new_skel = (new_skel | (meanlines > .3))
plt.imshow(new_skel)
new_skel = tf.copy()
for i in range(5):
    meanlines = mean_colored_connections(new_skel, Fmax, double_connect=False)
    old_size = new_skel.sum()    
    new_skel = (new_skel | (meanlines > .3))
    print('added', new_skel.sum() - old_size, 'new pixels', '*'*30)
    print('*'*80)
reload(network_completion)
reload(network_completion)
reload(network_completion)
from network_completion import connect_iterative
connect_iterative(tf, F[:-4].max(axis=0))
reload(network_completion)
from network_completion import connect_iterative
connect_iterative(tf, F[:-4].max(axis=0))
ci = _
ci = _402
plt.imshow(ci)
plt.imshow(plt.maximum(ci,tf*2))
plt.imshow(np.maximum(ci,tf*2)[crop])
plt.imshow(np.maximum(ci*1.,tf*2.)[crop])
reload(network_completion)
from network_completion import connect_iterative
connect_iterative(tf, F[:-4].max(axis=0))
iterative_network = _
iterative_network = _411
plt.imshow(iterative_network[crop])
plt.imsave('iterative_network', iterative_network[crop])
get_ipython().run_line_magic('save', 'line_to_nearest_mash.py 1-416')
