#!/usr/bin/env python3

"""
this is to convert the RGB colored manual traces into usuable traces
with pixel widths instead of colors
"""

# T is the RGB_TRACE

# the colors corresponding to each pixel width
COLORD = {
    3: "#ff006f", # magenta
    5: "#a80000", # dark red
    7: "#a800ff", # purple
    9: "#ff00ff", # light pink
    11: "008aff", # blue
    13: "8aff00", # green
    15: "ffc800", # dark yellow
    17: "ff8a00", # orange
    19: "ff0015"  # bright red
}

def hex_to_rgb(hexstring):
    """
    there's a function that does this in matplotlib.colors
    but its scaled between 0 and 1 but not even as an
    array so this is just as much work
    """
    triple = hexstring.strip("#")
    return tuple(int(x,16) for x in (triple[:2],triple[2:4],triple[4:]))

# a 2D picture to fix in with the pixel widths
widthtrace = np.zeros_like(T[:,:,0])

for pix, color in COLORD.items():
    # get a (rr,gg,bb) triple
    triple = hex_to_rgb(color)

    # get the 2D indices that are that color
    idx = np.where(np.all(img == triple, axis=-1))
    widthtrace[idx] = pix

# then save the file



