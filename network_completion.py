from skimage.filters.rank import sum as local_count
from skimage.draw import line
import numpy as np
from itertools import combinations

def find_endpoints(arr):
    """Find endpoints of a boolean array (like a thinned approximation)

    Will return any pixels that have between 1 and 2 adjacent neighbors using
    2-connectivity. 2 neighbors means it's directly connected to one other
    pixel, and 1 pixel means it's not connected to anything else.
    """
    # 2-connectivity
    selem = np.ones((3,3), np.bool)

    a = arr.astype('bool')

    neighbors = local_count(arr.astype('uint8'), selem)

    return arr & ((neighbors==1)|(neighbors==2))

def categorize_endpoints(arr):
    """Find endpoints, as in find_endpoints, but label each with the *kind*
    of endpoint it is describing its orientation. Is it 2 connected by:

        _ _ _ <--- top (0)
        _ . _ <--- middle (1)
        _ _ _ <--- bottom (2)
        ^ ^ ^
        | | |_______ right (0)
        | |_________ middle (1)
        |___________ left (2)

    these are the same as the row/column indices into the 3x3 square. so
    if a pixel's only neighboring pixel is to its immediate right, then
    the label for the pixel is (1,0). If the endpoint is isolated (i.e. it
    has no neighbors) then its label is (1,1) (since 1 is the more forgivable
    connection condition).

    The idea is that
    -   a left-connected pixel should not connect with another left-connected
        pixel
    -   a right-connected pixel should not connect with another right-connected
        pixel.
    -   a bottom-connected pixel should not be connected with another
        bottom-connected pixel
    -   a top-connected pixel should not be connected with another
        top-connected pixel

    There are other ideas (that a connection shouldn't be made through a point
    where the frangi score is 0; that a connection shouldn't be made that
    crosses an established part of the skeleton; that connections shouldn't
    be made between two endpoints whose distance exceeds some specified amount)
    that cannot be cast in terms of these labels, but this does reduce the
    number of possible connections substantially.

    this returns two lists of tuples, where the first contains the indices of
    each endpoint, and the corresponding place in the second list contains
    its label (also a tuple)
    """

    # still get a list of endpoints because this is faster.
    selem = np.ones((3,3), np.bool)
    a = arr.astype('bool')
    endpoint_list = list()
    label_list = list()
    bd = np.array([[1,1,1],[1,0,1],[1,1,1]],dtype='bool')
    for px, py in zip(*np.where(a)):
        local = a[px-1:px+2,py-1:py+2]

        if local.sum() == 1:
            endpoint_list.append((px,py))
            label_list.append((1,1))
        elif local.sum() == 2:
            endpoint_list.append((px,py))
            label_list.append(tuple(map(int, np.where(local&bd))))
        else:
            pass # this isn't an endpoint

    return endpoint_list, label_list


def mean_colored_connections(arr, scores=None, double_connect=True):
    """Lines that connect nearest endpoints, colored by the mean value of
        that line through the matrix scores

        if scores is None, just return a boolean array
    """
    # list of endpoints as ordered pairs
    a = arr.astype('bool') # cast to bool first to prevent weirdness
    neighbors = local_count(a.astype('uint8'), np.ones((3,3), np.bool))
    endpoints = a & ((neighbors==1)|(neighbors==2))
    endpoint_list = list(zip(*np.where(endpoints)))

    lines = np.zeros(endpoints.shape) # but type float
    for p0 in endpoint_list:
        if (neighbors[p0] == 2) or not double_connect:
            nearest = endpoint_list[np.argmin([
                        (p0[0]-p[0])**2 + (p0[1] - p[1])**2
                        if p0!=p else 10000 for p in endpoint_list]
                        )]
            #print(p0, nearest)
            line_to_add = line(*p0, *nearest)

            if scores is None:
                lines[line_to_add] = 1
            else:
                lines[line_to_add] = scores[line_to_add].mean()
                #print(scores[line_to_add].mean())
        else:
            # same thing but get the two nearest
            nearest, second, *__ = np.argpartition([
                        (p0[0]-p[0])**2 + (p0[1] - p[1])**2
                        if p0!=p else 10000 for p in endpoint_list], 2)
            nearest = endpoint_list[nearest]
            second = endpoint_list[second]
            print(p0, nearest, second, '<---------------')
            line_to_add = line(*p0, *nearest)
            line2_to_add = line(*p0, *second)
            if scores is None:
                lines[line_to_add] = 1
                lines[line2_to_add] = 1
            else:
                lines[line_to_add] = scores[line_to_add].mean()
                lines[line2_to_add] = scores[line2_to_add].mean()
                #print(scores[line_to_add].mean())
                #print(scores[line2_to_add].mean())

    return lines


def connect_iterative(arr, scores=None, max_iterations=10):
    """Lines that connect nearest endpoints, colored by the mean value of
        that line through the matrix scores

        if scores is None, just return a boolean array
    """
    # list of endpoints as ordered pairs
    a = arr.astype('bool') # cast to bool first to prevent weirdness
    bad_pairs = list()
    lines = a.copy() # but type float
    for i in range(max_iterations):
        neighbors = local_count(a.astype('uint8'), np.ones((3,3), np.bool))
        endpoints = a & ((neighbors==1)|(neighbors==2))
        endpoint_list = list(zip(*np.where(endpoints)))
        print(f"at start of {i}th iteration, {len(bad_pairs)} many bad pairs")
        old_network_size = lines.sum()
        for k, p0 in enumerate(endpoint_list):
            candidates = [p for p in endpoint_list[k:]
                          if ((p,p0) not in bad_pairs and
                              (p0,p) not in bad_pairs)]
            if not candidates:
                continue
            nearest = candidates[np.argmin([
                            (p0[0]-p[0])**2 + (p0[1] - p[1])**2
                            if p0!=p else 10000 for p in candidates]
                            )]
            line_to_add = line(*p0, *nearest)

            if scores is None:
                lines[line_to_add] = 1
            else:
                mean_score = scores[line_to_add].mean()
                if (mean_score < .3) or (scores[line_to_add]==0).any():
                    bad_pairs.append((p0,nearest))
                    continue
                else:
                    lines[line_to_add] = 1
                #print(scores[line_to_add].mean())
        new_network_size =lines.sum()
        size_diff =  new_network_size - old_network_size
        print(f'after {i}th iteration,', size_diff, 'pixels were added')
        if size_diff == 0:
            break
    else:
        print(f'returned after {max_iterations} iterations')

    return lines


def _euclidean_distance(p0,p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)


def connect_iterative_by_label(arr, scores, max_iterations=10,
                               max_dist=None):
    """Lines that connect nearest endpoints, colored by the mean value of
        that line through the matrix scores

        if scores is None, just return a boolean array
    """
    # list of endpoints as ordered pairs
    a = arr.astype('bool') # cast to bool first to prevent weirdness
    matched = list()


    endlist, endlabs = categorize_endpoints(arr)
    N_ends = len(endlist)
    # handshake matrix, matchable[j,k]==0 if endlist[j] and endlist[k]
    # cannot be connected, otherwise 1. this is upper triangular
    # on the first diagonal, so make sure k > j
    matchable = np.triu(np.ones((len(endlist), len(endlist))), k=1)

    # distances between these two points with lazy filling. could be overloaded
    # into matchable but let's keep it separate
    dists = np.zeros(matchable.shape, dtype=np.float64)
    print(f'there are {len(endlist)} endpoints')
    
    print('checking compatibility of endpoints by labels and calculating distances')
    
    for j, jlab in enumerate(endlabs):

        # still calculate the distances below
        #if jlab == (1,1):
        #    continue # this can connect with anything

        for k, klab in enumerate(endlabs[j+1:],j+1):
            if jlab[0] != 1:
                if jlab[0] == klab[0]:
                    matchable[j,k] = 0
                    continue
            if jlab[1] != 1:
                if jlab[1] == klab[1]:
                    matchable[j,k] = 0
                    continue
            # if they are matchable, let's find the distance between them
            dist = _euclidean_distance(endlist[j], endlist[k])
            if max_dist is not None:
                if dist > max_dist:
                    matchable[j,k] = 0
                else:
                    dists[j,k] = dist
            else:
                dists[j,k] = dist


    lines = a.copy() # but type float

    print(f"removed {N_ends*(N_ends-1)/2 - matchable.sum():n} "
           "pairs of endpoints from consideration "
          f'(out of {N_ends*(N_ends-1) / 2:n} possible pairs)')

    size_before = matchable.sum()    
    for j, pj in enumerate(endlist):
        for k, pk in enumerate(endlist[j+1:], j+1):
            if not matchable[j,k]:
                continue
            l = line(*pj, *pk)
            # lines will have the two endpoints so sum is at least 2
            if (lines[l].sum() > 2) or (scores[l]==0).any():
                matchable[j,k] = 0
                dists[j,k] = 0
    
    size_after = matchable.sum()

    print(f"removed {size_before-size_after} more pairs.")
    
    for j, pj in enumerate(endlist):
        mean_score = lambda p: scores[line(*pj,*p)].mean()
        point_scores = [(k, pk, mean_score(pk))
                        for k,pk in enumerate(endlist[j+1:], j+1)
                        if matchable[j,k]]
        if not point_scores:
            continue
        s = sorted(point_scores, key=lambda x: x[2])
        p_best = s[-1][1]
        lines[line(*pj, *p_best)] = 1

    return lines
    #for i in range(max_iterations):
    #    old_network_size = lines.sum()

    #    print(f"at start of {i}th iteration:")
    #    print(f"\t{matchable.any(axis=1).sum()} unmatched endpoints")
    #    print(f"\t{old_network_size} is the size of the network")

    #    for j, p0 in enumerate(endlist):

    #        if not matchable[k,:].any():
    #            continue # we can't connect this endpoint or it's already
    #                     # been connected to

    #        for k, p in enumerate(endlist[j+1:],j+1):
    #            if not matchable[j,k]:
    #                continue

    #        if not candidates:
    #            continue # this endpoint can't connect to anything anymore

    #        for candidate in candidate:
    #            pass
    #        #nearest = candidates[np.argmin([
    #        #                (p0[0]-p[0])**2 + (p0[1] - p[1])**2
    #        #                if p0!=p else 10000 for p in candidates]
    #        #                )]
    #        #line_to_add = line(*p0, *nearest)

    #        #if scores is None:
    #        #    lines[line_to_add] = 1
    #        #else:
    #        #    mean_score = scores[line_to_add].mean()
    #        #    if (mean_score < .3) or (scores[line_to_add]==0).any():
    #        #        bad_pairs.append((p0,nearest))
    #        #        continue
    #        #    else:
    #        #        lines[line_to_add] = 1
    #        #    #print(scores[line_to_add].mean())
    #    new_network_size = lines.sum()
    #    size_diff =  new_network_size - old_network_size
    #    print(f'after {i}th iteration,', size_diff, 'pixels were added')
    #    if size_diff == 0:
    #        break
    #    else:
    #        old_network_size = new_network_size
    #else:
    #    print(f'returned after {max_iterations} iterations')

    #return lines


def colored_connections_any_nonzero(arr, scores):
    """Lines that connect nearest endpoints, colored by the mean value of
        that line through the matrix scores

        WOULD BE BETTER IF YOU USED SCORES FROM THE APPROPRIATE SCALE
        if scores is None, just return a boolean array
    """
    # list of endpoints as ordered pairs
    a = arr.astype('bool') # cast to bool first to prevent weirdness
    neighbors = local_count(a.astype('uint8'), np.ones((3,3), np.bool))
    endpoints = a & ((neighbors==1)|(neighbors==2))
    endpoint_list = list(zip(*np.where(endpoints)))

    lines = np.zeros(endpoints.shape) # but type float
    #good_lines = list()
    for i, p0 in enumerate(endpoint_list):
        for j, p1 in enumerate(endpoint_list[i+1:], i+1):

            #nearest = endpoint_list[np.argmax([
            #            (p0[0]-p[0])**2 + (p0[1] - p[1])**2
            #            if p0!=p else 10000 for p in endpoint_list]
            #            )]

            candidate = line(*p0, *p1)

            if (scores[candidate] > 0).all():
                lines[candidate] = scores[candidate].mean()

    return lines


def colored_connections_max_path(arr, scores):
    """Lines that connect nearest endpoints, colored by the mean value of
        that line through the matrix scores

        WOULD BE BETTER IF YOU USED SCORES FROM THE APPROPRIATE SCALE
        if scores is None, just return a boolean array
    """
    # list of endpoints as ordered pairs
    a = arr.astype('bool') # cast to bool first to prevent weirdness
    neighbors = local_count(a.astype('uint8'), np.ones((3,3), np.bool))
    endpoints = a & ((neighbors==1)|(neighbors==2))
    endpoint_list = list(zip(*np.where(endpoints)))

    lines = np.zeros(endpoints.shape) # but type float
    #good_lines = list()
    for i, p0 in enumerate(endpoint_list):
        mean_score = lambda p: scores[line(*p0,*p)].mean() if ((p0!=p) and np.all(scores[line(*p0, *p)] >0)) else 0
        point_scores = [(p, mean_score(p)) for p in endpoint_list[i+1:]]
        point_scores = [(p, score) for p, score in point_scores if score!=0]
        s = sorted(point_scores, key=lambda x: x[1])
        if i<5:
            print(f'{i}:', s)
        for p, score in s:
            path = line(*p0, *p)
            lines[path] = score

    return lines

if __name__ == "__main__":



    import matplotlib.pyplot as plt
    from skimage.util import img_as_float
    from skimage.io import imread
    from placenta import (get_named_placenta, list_by_quality, cropped_args,
                        mimg_as_float)

    from frangi import frangi_from_image
    import numpy.ma as ma
    from hfft import fft_gradient, fft_hessian, fft_gaussian
    from merging import nz_percentile
    from plate_morphology import dilate_boundary
    import os.path, os
    from skimage.morphology import thin
    from postprocessing import dilate_to_rim
    from scoring import confusion, mcc
    from placenta import open_tracefile, open_typefile  

    filename = list_by_quality(N=1)[0]
    name_stub = filename.rstrip('.png').strip('T-')
    img = get_named_placenta(filename)
    cimg = open_typefile(filename, 'raw')
    crop = cropped_args(img)
    trace = open_tracefile(filename) 
    scales = np.logspace(-1.5, 3.5, num=20, base=2)
    threshold = .15
    neg_threshold = .001

    F = np.stack([frangi_from_image(img, sigma, beta=0.15, gamma=1.0, dark_bg=False,
                                    signed_frangi=True, dilation_radius=20,
                                    rescale_frangi=True)
                                    for sigma in scales])
    # need to fix this in the signed_frangi logic
    F = F*~dilate_boundary(None, mask=img.mask, radius=20)
    
    Fmax = (F*(F>0))[:-6].max(axis=0)
    Fneg = -(F*(F<0))[:2].min(axis=0)

    approx = Fmax > threshold
    rim_approx = (Fneg > neg_threshold)
    skel = thin(approx)
    completed = connect_iterative_by_label(skel, Fmax, max_dist=100) 
    completed_dilated = dilate_to_rim(completed, rim_approx, max_radius=10)
    approx_dilated = dilate_to_rim(approx, rim_approx, max_radius=10)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
    A = ax.ravel()
    

    precision = lambda t: int(t[0]) / int(t[0] + t[2])

    mcc_FA, counts_FA = mcc(approx, trace, bg_mask=img.mask, return_counts=True)
    mcc_FAD, counts_FAD = mcc(approx_dilated, trace, bg_mask=img.mask, return_counts=True)
    mcc_NCD, counts_NCD = mcc(completed_dilated, trace, bg_mask=img.mask, return_counts=True)

    prec_FA = precision(counts_FA)
    prec_FAD = precision(counts_FAD)
    prec_NCD = precision(counts_NCD)
    
    A[0].imshow(cimg[crop])
    A[0].set_title(name_stub)
    A[1].imshow(confusion(approx, trace, bg_mask=img.mask)[crop])
    A[1].set_title(fr'    fix $\alpha={threshold:.2f}$', loc='left')
    A[1].set_title(f'MCC: {mcc_FA:.2f}\n'
                   f'precision: {prec_FA:.2%}', loc='right')

    A[2].imshow(confusion(approx_dilated, trace, bg_mask=img.mask)[crop])
    A[2].set_title(fr'    dilate_to_rim $\alpha={threshold:.2f}$', loc='left')
    A[2].set_title(f'MCC: {mcc_FAD:.2f}\n'
                   f'precision: {prec_FAD:.2%}', loc='right')

    A[3].imshow(confusion(completed_dilated, trace, bg_mask=img.mask)[crop])
    A[3].set_title(fr'    network_completed, dilated', loc='left')
    A[3].set_title(f'MCC: {mcc_NCD:.2f}\n'
                   f'precision: {prec_NCD:.2%}', loc='right')
    [a.axis('off') for a in A]
    plt.tight_layout()
