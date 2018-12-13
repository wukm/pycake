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


