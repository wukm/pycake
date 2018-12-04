#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def _bunch_hists(H, bunches):

    return np.stack((np.sum(np.atleast_2d(H[b,:]), axis=0) for b in bunches))


def scale_to_width_plots(multiscale_approx, max_labels, widths, scales,
                         bunches=None, cmap=None, approx_method=None):
    """
    multiscale_approx is a 3d boolean array whose first dimension is scale
    max_labels is a 2d array of integers that say where the max value of
    F occured. you can get max_labels by running V.argmax(axis=0)

    in widths, each pixel has a unique width

    bunches.flatten() should be the same as arange(scales)
    but can be something like
    ( (0,1,2,3), (4,5), 6, 7, 8, (9,10,11) )

    or even

    (2,3,4,5,(0,1,6,7))

    this is to prevent similar scales from clogging, you can just bin them
    all together.

    approx method is a label to use in the fig titles
    """
    plt.style.use('seaborn')

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,20))

    A = multiscale_approx  # easier to work with

    wbins = np.arange(3,20,2)  # bins of widths in ground truth

    max_hists = [[np.sum((max_labels == s) & (widths==w)) for w in wbins]
                  for s in range(len(scales))]

    hists = np.array([[np.sum((widths==w) & A[n]) for w in wbins]
                      for n in range(len(scales))]
                     )


    if cmap is None:
        # this will just use the default cycle of colors
        colors = np.repeat(None, len(scales))
    else:
        if not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            # try this
            cmap = plt.get_cmap(cmap)

        colors = cmap(np.linspace(0,1,len(scales)))

    labels = [rf'$\sigma_{{{k}}}={sigma:.2f}$'
              for k, sigma in enumerate(scales,1)]

    # number of true positives, false negatives for each width
    tp_hists = [np.sum((widths==w) & A.any(axis=0)) for w in wbins]
    fn_hists = [np.sum((widths==w) & ~A.any(axis=0)) for w in wbins]


    if bunches is not None:
        hists = _bunch_hists(hists, bunches)

        # just return \sigma_{1,2,3} or something rather than listing
        bunch_label = lambda b: r"$\sigma_{{{}}}$".format(','.join(map(str,b)))

        labels = [label if np.isscalar(b) else bunch_label(b)
                  for b, label in zip(bunches, labels)]

        colors = [color if np.isscalar(b) else colors[np.median(b)]
                  for b, color in zip(bunches, colors)]

    ax[0].bar(wbins, tp_hists, color=(0.6,0.6,0.6),
            label='# true positives')
    ax[0].bar(wbins, fn_hists, bottom=tp_hists, color=(1,.4,.4,1),
            label='# false negatives')

    for h, mh, label, color in zip(hists, max_hists, labels, colors):
        ax[0].plot(wbins, h, label=label, color=color, markersize=1.5,
                   linewidth=1.2)
        ax[1].plot(wbins, mh, label=label, color=color, markersize=1.5,
                   linewidth=1.2)


    ax[0].set_xticks(wbins)
    ax[0].set_xlabel('vessel widths (ground truth), pixels')
    ax[0].set_ylabel('# pixels')

    ax[1].set_xticks(wbins)
    ax[1].set_xlabel('vessel widths (ground truth), pixels')
    ax[1].set_ylabel('# pixels')

    title = 'pixels reported per scale'
    max_title = r'pixel widths of true positives by scale of $V_\max$'
    if approx_method is not None:
        title += f'({approx_method})'
        max_title += f'({approx_method})'

    ax[0].legend()
    ax[1].legend()

    return fig, ax
