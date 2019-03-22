# coding: utf-8
import numpy as np
import matplotlib as pyplot
import matplotlib.pyplot as plt
a = np.zeros((3,3), np.bool)
a
fig, ax = plt.subplots(nrows=3, ncols=3)
A = ax.ravel()
for n, axis in A.ravel():
    im = a.copy()
    im[1,1]=1
    im[n] = 1
    axis.imshow(im)
    axis.axis('off')
    
for n, axis in A:
    im = a.copy()
    im[1,1]=1
    im[n] = 1
    axis.imshow(im)
    axis.axis('off')
    
fig, ax = plt.subplots(nrows=3, ncols=3)
ax
ax.ravel()
ax
for n, axis in ax.ravel():
    im = a.copy()
    im[1,1]=1
    im[n] = 1
    axis.imshow(im)
    axis.axis('off')
    
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im[1,1]=1
    im[n] = 1
    axis.imshow(im)
    axis.axis('off')
    
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im[1,1]=1
    im = 1
    axis.imshow(im)
    axis.axis('off')
    
get_ipython().run_line_magic('pinfo', 'numpy.put')
get_ipython().run_line_magic('pinfo', 'np.put')
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n) = 1
    im[1,1] = 1
    axis.imshow(im)
    axis.axis('off')    
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im)
    axis.axis('off') 
       
plt.show()
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.matshow(im)
    axis.axis('off') 
    
       
plt.show()
plt.show()
fig, ax = plt.subplots(nrows=3, ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.matshow(im)
    axis.axis('off') 
    
plt.show()
help(plt.matshow)
fig, ax = plt.subplots(nrows=3, ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    axis.axis('off') 
    
    
fig, ax = plt.subplots(nrows=3, ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    axis.axis('off') 
    axis.set_title()
    
    
get_ipython().run_line_magic('pinfo', 'np.unravel_index')
np.unravel_index(6, (3,3))
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    axis.axis('off') 
    axis.set_title(np.unravel_index(n,(3,3)))
        
plt.show()
axis.set_frame_on()
axis.set_frame_on(True)
fig, ax = plt.subplots(nrows=3,ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    axis.axis('off') 
    axis.set_title(np.unravel_index(n,(3,3)))
    axis.set_frame_on(True)
            
plt.show()
fig, ax = plt.subplots(nrows=3,ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    #axis.axis('off') 
    axis.set_title(np.unravel_index(n,(3,3)))
    axis.set_frame_on(True)
            
plt.show()
axis.set_xticklabels([])
fig, ax = plt.subplots(nrows=3,ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray)
    #axis.axis('off') 
    axis.set_title(np.unravel_index(n,(3,3)))
    axis.set_frame_on(True)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
             
plt.show()
fig, ax = plt.subplots(nrows=3,ncols=3)
for n, axis in enumerate(ax.ravel()):
    im = a.copy()
    im.put(n,1)
    im[1,1] = 1
    axis.imshow(im, cmap=plt.cm.gray_r)
    #axis.axis('off') 
    axis.set_title(np.unravel_index(n,(3,3)))
    axis.set_frame_on(True)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
             
plt.show()
