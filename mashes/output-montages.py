# coding: utf-8
import os
stub = T-BN2050224
stub = 'T-BN2050224'
filename = stub+'.png'
get_ipython().run_line_magic('cd', '..')
from placenta import crop, open_typefile
from placenta import cropped_args, open_typefile
from placenta import cropped_args, open_typefile, get_named_placenta
img = get_named_placenta(filename)
crop = cropped_args(img)
crop
ctrace = open_typefile(filename, 'ctrace')
ctrace
os.listdir('examples/')
figs = [f for f in os.listdir('examples/') if f.startswith(stub)]
figs
help(os.listdir)
figs = ['examples/'+f for f in os.listdir('examples/') if f.startswith(stub)]
figs
from skimage.io import imread
from skimage.misc import montage
from skimage.util import montage
import numpy as np
import matplotlib.pyplot as plt
confusion_pf = imread(figs[1])
labeled_FA = imread(figs[2])
confusion_FA = imread(figs[3])
confusion_S = imread(figs[4])
confusion_rw = imread(figs[7])
labeled_rw = imread(figs[6])
labeled_pf = imread(figs[8])
labeles_S = imread(figs[9])
labeled_S = imread(figs[9])
fmax = imread(figs[10])
del labeles_pf
del labeles_S
np.who()
get_ipython().run_line_magic('ls', '')
raw = open_typefile(filename, 'raw')
figs
imgpp = imread(figs[0])
np.who()
raw[crop]
get_ipython().run_line_magic('ls', '')
np.who
np.who()
get_ipython().run_line_magic('pinfo', 'raw.expand_dims')
np.stack((raw, imgpp[:3], ctrace, fmax[:3]))
np.stack((raw, imgpp[...,:3], ctrace, fmax[...,:3]))
imgpp[...,:3]
_.shape
np.stack((raw[crop], imgpp[...,:3], ctrace[crop], fmax[...,:3]))
montage(_, multichannel=True)
plt.imshow(_)
plt.show()
M1 = montage(np.stack((raw[crop], imgpp[...,:3], ctrace[crop], fmax[...,:3])), multichannel=True)
M2 = montage(np.stack((labeled_pf, labeled_FA, labeled_rw, labeled_S,
confusion_pf, confusion_FA, confusion_rw, confusion_S)))
M2 = montage(np.stack((labeled_pf, labeled_FA, labeled_rw, labeled_S,
confusion_pf, confusion_FA, confusion_rw, confusion_S)), multichannel=True)
M2
_.shape
plt.imshow(M2)
plt.show()
help(montage)
M2 = montage(np.stack((labeled_pf, labeled_FA, labeled_rw, labeled_S,
confusion_pf, confusion_FA, confusion_rw, confusion_S)), multichannel=True, grid_shape=(4,2))
plt.imshow(M2)
plt.show()
M2 = montage(np.stack((labeled_pf, labeled_FA, labeled_rw, labeled_S,
confusion_pf, confusion_FA, confusion_rw, confusion_S)), multichannel=True, grid_shape=(2,4))
plt.imshow(M2)
plt.show()
plt.imsave('M2-'+filename, M2)
plt.imsave('M1-'+filename, M1)
M1 = montage(np.stack((raw[crop], imgpp[...,:3], ctrace[crop], fmax[...,:3])), multichannel=True, grid_shape=(1,4))
plt.imshow(M1)
plt.show()
plt.imsave('M1-'+filename, M1)
