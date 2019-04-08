#!/usr/bin/env python3

"""
troughplot.py
The goal is to create just a single non-straight curve that is actually visually
troughlike i.e. it has a geometry of a semicircular trough. this is simply
for a visual representation of the sample problem


TODO: catch runtime error with imaginaries in
    Z = np.sqrt(r**2 - X**2)

    make it look a little nicer
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

sns.set()

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-2, 2, .05)
Y = np.arange(-2, 2, .05)

n_theta = 50
n_phi = 200

r = 1
r = 1 # radius of trough
X, Y = np.meshgrid(X,Y)

Z = np.sqrt(r**2 - X**2)

Z[X < -r] = 0
Z[X > r] = 0


surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, vmin=0, vmax=1.5,
                       cmap=cm.viridis, linewidth=0.1)

ax.set_autoscalez_on('False')
ax.set_zlim(-1, 1)

# make it like an image, not a graph
ax.set_zticklabels("")
ax.set_yticklabels("")
ax.set_xticklabels("")

