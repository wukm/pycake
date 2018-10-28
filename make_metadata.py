#!/usr/bin/env python3

""" one time use script to decode the filenames of the NCS samples
    into a json file with metadata, so you could, for example,
    get all placentas that were prepared by a certain person or
    that were prepared before a certain date or something

basicaly is to decode the filenames which are like
T-BN0013990_fetalsurface_fixed_ruler_lights_filter_12_0130-dd

into
    fullname: T-BN0013990_fetalsurface_fixed_ruler_lights_filter_12_0130-dd
    id: T-BN0013990
    date: 120130
    prepid: dd

ideally there would also be info about the image metadata
as in resolution, etc.
"""

pass
