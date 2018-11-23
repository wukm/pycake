# coding: utf-8
Afile = 'T-BN5852719.png'
A = get_named_placenta(Afile)
ucip = open_typefile(Afile, 'ucip')
cutmarks = np.nonzero(np.all(ucip==(0,0,255), axis=-1))
X, Y = cutmarks[0][0], cutmarks[1][0]
threshold = img[cutmarks].mean() * .85
cutregion = np.s_[X-100:X+100, Y-100:Y+100]
markers = np.zeros(img.shape, dtype='int32')
markers[img.filled(255) < threshold] = 2
markers[img.mask] = 1
markers[cutmarks] = 1
cutfix = watershed(img.filled(255) < threshold, markers=markers)
new_mask = img.mask.copy()
new_mask[cutregion] = cutfix[cutregion]
plt.imshow(new_mask)
plt.show()
threshold
plt.imshow(img[cutregion])
plt.show()
plt.imshow(img)
plt.show()
threshold = A[cutmarks].mean()*.85
markers = np.zeros(A.shape, dtype='int32')
markers[A.filled(255) < threshold] = 2
markers[A.mask] = 1
markers[cutmarks] 1
markers[cutmarks] = 1
cutfix = watershed(img.filled(255) < threshold, markers=markers)
cutfix = watershed(A.filled(255) < threshold, markers=markers)
new_mask = A.mask.copy()
new_mask[cutregion] = cutfix[cutregion]
plt.imshow(new_mask)
plt.show()
plt.imshow(cutfix)
plt.show()
new_mask = A.mask.copy()
new_mask[cutregion] = np.invert(cutfix[cutregion])
plt.imshow(new_mask)
plt.show()
plt.imshow(cutfix)
plt.show()
new_mask = A.mask.copy()
plt.imshow(new_mask)
plt.show()
new_mask[cutregion]
plt.imshow(_)
plt.show()
plt.imshow(cutfix[cutregion])
plt.show()
plt.imshow(cutfix[cutregion] == 1)
plt.show()
new_mask[cutregion] = cutfix[cutregion] == 1
plt.imshow(new_mask)
