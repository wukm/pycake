#!/usr/bin/env python

"""
This should be a plugin to take images from the folder NCS_vessel_GIMP_xcf
and create trace, mask, and backgrounded images from each xcf file.

chmod +x and then copy or link to  ~/gimp-2.x/plug-ins/
"""

from gimpfu import *
import os.path
from functools import partial

#basefile, ext = os.path.splitext(xcffile)

def _outname(base, s=None):
    if s is None:
        stubs = (base, 'png')
    else:
        stubs = (base, s, 'png')

    return '.'.join(stubs)

# get active image
def process_NCS_xcf(timg,tdrawable):
    img = timg
    basename, _ = os.path.splitext(img.name)

    # generate output names easier
    outname = partial(_outname, base=basename)

    # get coordinates of the center
    cx, cy = img.height // 2 , img.width // 2

    # disable the undo buffer
    #img.disable_undo()

    #perimeter = pdb.gimp_image_get_layer_by_name(img, 'perimeter')

    for layer in img.layers:
        if layer.name.lower() in ('perimeter', 'perimeters'):
            # .copy() has optional arg of "add_alpha_channel"
            mask = layer.copy()

        layer.visible = False

    mask.name = "mask" # name the new layer
    img.add_layer(mask,0) # add in position 0 (top)

    pdb.gimp_layer_flatten(mask) # Remove Alpha Channel.

    # remove unneeded (i hope) annotations
    # color exchange yellow & blue to black
    pdb.plug_in_exchange(img,mask,255,255,0,0,0,0,1,1,1)
    pdb.plug_in_exchange(img,mask,0,0,255,0,0,0,1,1,1)

    # set FG color to black (for tools, not of image)
    gimp.set_foreground(0,0,0)


    # Bucket Fill Inside black (center pixel is hopefully fine)
    pdb.gimp_edit_bucket_fill(mask,0,0,100,0,0,cx,cy)

    # Color Exchange Green to White.
    pdb.plug_in_exchange(img,mask,0,255,0,255,255,255,1,1,1)

    # Color Exchange Cyan (00ffff) to White.
    pdb.plug_in_exchange(img,mask,0,255,255,255,255,255,1,1,1)

    # Export Layer as Image called "f".mask.png
    pdb.gimp_file_save(img,mask, outname(s="mask"), '')

    # invert (so exterior is now black)
    pdb.gimp_invert(mask)
    mask.mode = DARKEN_ONLY_MODE # the constant 9

    # set bottom layer (placenta) to visible
    raw = img.layers[-1]
    raw.visible = True

    # now make a new layer called 'raw_img' from visible
    base = pdb.gimp_layer_new_from_visible(img,img,'base')
    img.add_layer(base,0)
    pdb.gimp_file_save(img , base, outname(s=None) , '')

    # now get rid of mask and save the raw image
    mask.visible = False
    pdb.gimp_file_save(img, raw, outname(s='raw') ,'')

    # now set the veins/artery layers as only visible
    for layer in img.layers:
        layer.visible = (layer.name.lower() in ("arteries", "veins"))

    trace = pdb.gimp_layer_new_from_visible(img,img,'trace')
    img.add_layer(trace,0)

    pdb.gimp_layer_flatten(trace) # remove alpha channel

    pdb.gimp_desaturate(trace) # turn to grayscale
    pdb.gimp_threshold(trace,255,255) # anything not 255 turns black

    pdb.gimp_file_save(img, trace, outname(s='trace') ,'')

register(
    "process_NCS_xcf",
    "Create base image + trace + mask from an NCS xcf file",
    "Create base image + trace + mask from an NCS xcf file",
    "Luke Wukmer",
    "Luke Wukmer",
    "2018",
    "<Image>/Image/Process_NCS_xcf...",
    "RGB*, GRAY*",
    [],
    [],
    process_NCS_xcf)

main()
