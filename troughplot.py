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

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-6,6, 0.05)
Y = np.arange(-6,6, 0.05)

r = 1 # radius of trough
X, Y = np.meshgrid(X,Y)

Z = np.sqrt(r**2 - X**2)

Z[X < -r] = 0
Z[X > r] = 0


surf = ax.plot_surface(X,Y,Z, cmap=cm.jet)

ax.set_autoscalez_on('False')
ax.set_zlim(-2, 5)

# make it like an image, not a graph
ax.set_zticklabels("")
ax.set_yticklabels("")
ax.set_xticklabels("")

