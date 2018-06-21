MSE
plt.show()
ms(A)
plt.show()
plt.imshow(A)
plt.show()
A.max()
img.mask
plt.imshow(img.mask)
plt.show()
from skimage.segmentation import mark_boundaries
mask
M = img.mask
plt.imshow(M)
plt.show()
mark_boundaries(M)
mark_boundaries(M,M)
plt.imshow(_)
plt.show()
plt.imshow(np.zeros_like(M),M)
mark_boundaries(np.zeros_like(M),M)
plt.imshow(_)
plt.show()
bound = mark_boundaries(np.zeros_like(M),M)
bound
bound.max()
from skimage.morphology import binary_dilate
from skimage.morphology import binary_dilation
from skimage.morphology import disk
binary_dilation(bound, selem=disk(2))
disk(2)
bound
bound.astype('b')
binary_dilation(bound.astype('b'), selem=disk(2))
help(binary_dilation)
binary_dilation(bound.astype('b'), selem=disk(3))
binary_dilation(bound.astype('b'))
plt.imshow(_)
plt.show()
bound
plt.imshow(bound)
plt.show()
plt.imshow(bound, cmap=plt.cm.gray)
plt.show()
plt.imshow(bound, cmap=plt.cm.gray)
plt.show()
bound
bound
plt.imshow(bound)
plt.show()
binary_dilation(bound)
plt.imshow(_)
plt.show()
binary_dilation(bound)
b = _
b.any()
b*255.
plt.imshow(_)
plt.show()
b.find(1)
b.argmax()
b.argmax()
b[2901]
b(2901)
b[2901]
b
b.max()
plt.imshow(b)
plt.show()
b.astype('uint8')
b.astype('uint8')*255
plt.imshow(_)
plt.show()
bound
bound.count()
bound.sum()
mask
m
M
np.bitwise_and?
np.logical_and?
np.logical_or(M,bound)
np.logical_or(M,b)
b
b.shape
mask
img.mask
img.mask
img.mask.shape
mark_boundaries(np.zeros_like(img.mask),img.mask)
mark_boundaries(np.zeros_like(img.mask),img.mask).shape
find_boundaries(np.zeros_like(img.mask),img.mask).shape
from skimage.segmentation import find_boundaries
find_boundaries(img.mask)
plt.imshow(_)
plt.show()
help(find_boundaries)
bounds
bounds = find_boundaries(img.mask)
bounds
dilated_border = binary_dilation(bounds, selem=disk(int(sigma)))
plt.imshow(dilated_border)
plt.show()
from score import confusion
confusion(dilated_border, bounds)
plt.imshow(_)
plt.show()
sigma
np.ceil(sigma)
new_mask = np.logical_or(mask, dilated_border)
new_mask = np.logical_or(img.mask, dilated_border)
plt.imshow(new_mask)
plt.show()
mask = img.mask
confusion(mask,new_mask)
plt.imshow(_)
plt.show()
dilated_border = binary_dilation(bounds, selem=disk(20))
plt.imshow(confusion(dilated_border, bounds))
plt.show()
plt.imsave?
plt.imsave('mask_errors.png', confusion(dilated_border, bounds))
%save
%save boundarycalcs
%save 1-125 boundarycalcs
%history -f boundarycalcs
