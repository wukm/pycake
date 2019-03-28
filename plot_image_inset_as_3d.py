#!/usr/bin/env python3

"""
This is a demo program to show a manually selected inset of the original
image as a 3D graph, so as to display its curvilinearity
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from placenta import (list_by_quality, get_named_placenta, open_typefile,
                      cropped_args)

from skimage.filters import gaussian
import numpy as np

sns.set()

filename = list_by_quality(N=1)[0]
img = get_named_placenta(filename)
crop = cropped_args(img)

img = img/255.


inset = img[330:400,300:400].filled(0); sigma = 3.2
#inset = img[800:900,700:820].filled(0); sigma = 0.4

# flip to match the angle we'll look at later
plt.imshow(np.flip(inset), cmap=plt.cm.gray)
plt.axis('off')
plt.show()

plt.close()

X,Y = np.mgrid[0:inset.shape[0], 0:inset.shape[1]]

fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, inset, rstride=1, cstride=1,
                linewidth=0, cmap=plt.cm.viridis)

ax.view_init(azim=178,elev=63)
fig.tight_layout()
plt.show()
plt.close()

ig = gaussian(inset, sigma=sigma)

plt.imshow(np.flip(ig), cmap=plt.cm.gray)
plt.axis('off')
plt.show()
plt.close()

fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, ig, rstride=1, cstride=1, linewidth=0,
                cmap=plt.cm.viridis)
ax.view_init(azim=178,elev=63)
fig.tight_layout()
plt.show()
