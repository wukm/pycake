#!/usr/bin/env python

"""
This should be a plugin to take images from the folder NCS_vessel_GIMP_xcf
and create trace, mask, and backgrounded images from each xcf file.

to use:
chmod +x and then copy or link to  ~/gimp-2.x/plug-ins/
"""

from gimpfu import *
import os.path
from functools import partial

#basefile, ext = os.path.splitext(xcffile)

def _outname(base, s=None):

    #base = base.split("_", maxsplit=1)[0]
    if s is None:
        stubs = (base, 'png')
    else:
        stubs = (base, s, 'png')
    file
    filename = '.'.join(stubs)

    return os.path.join(os.getcwd(), filename)

# get active image
def process_NCS_xcf(timg,tdrawable):
    img = timg
    basename, _ = os.path.splitext(img.name) # split off extension .xcf
    basename = basename.split("_")[0] # only get T-BN-kjlksf part
    print "*"*80
    print '\n\n'

    print "Processing " , img.name
    # generate output names easier
    outname = partial(_outname, base=basename)

    # get coordinates of the center
    cx, cy = img.height // 2 , img.width // 2

    # disable the undo buffer
    img.disable_undo()

    #perimeter = pdb.gimp_image_get_layer_by_name(img, 'perimeter')

    for layer in img.layers:
        if layer.name.lower() in ('perimeter', 'perimeters'):
            # .copy() has optional arg of "add_alpha_channel"
            mask = layer.copy()
            break
    else:
        print "Could not find a perimeter layer."
        print "Layers of this image are:"
        for n,layer in enumerate(img.layers):
            print "\t", n, ":", layer.name
        print "Skipping this file."

        return

    for layer in img.layers:
        layer.visible = False

    mask.name = "mask" # name the new layer
    img.add_layer(mask,0) # add in position 0 (top)

    pdb.gimp_layer_flatten(mask) # Remove Alpha Channel.

    # save the annotated perimeter file (for calculations later)
    pdb.gimp_file_save(img,mask, outname(s="ucip"), '')

    # remove unneeded annotations from mask layer
    # color exchange yellow & blue to black
    pdb.plug_in_exchange(img,mask,255,255,0,0,0,0,1,1,1)
    pdb.plug_in_exchange(img,mask,0,0,255,0,0,0,1,1,1)

    # set FG color to black (for tools, not of image)
    gimp.set_foreground(0,0,0)

    # Bucket Fill Inside black (center pixel is hopefully fine,
    # do rest manually
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
    pdb.gimp_file_save(img, base, outname(s='raw') ,'')


    # now make the other one visible (this is dumb)
    for layer in img.layers:
        if layer.name.lower() in ("arteries", "veins"):
            layer.visible = True
        else:
            layer.visible = False
    # now with these two visible, merge them and add layer
    trace = pdb.gimp_layer_new_from_visible(img,img,'trace')
    img.add_layer(trace,0)

    pdb.gimp_layer_flatten(trace) # remove alpha channel

    # don't turn binary anymore
    #pdb.gimp_desaturate(trace) # turn to grayscale
    #pdb.gimp_threshold(trace,255,255) # anything not 255 turns black

    pdb.gimp_file_save(img, trace, outname(s='trace') ,'')

    # now extract an each type individually.
    found = 0
    for subtype in ("arteries", "veins"):
        for layer in img.layers:
            if layer.name.lower() == subtype:
                layer.visible = True
                pdb.gimp_layer_flatten(layer) # remove alpha channel
                pdb.gimp_file_save(img, layer, outname(s=subtype), '')
                layer.mode = 9 # set to darken only (for merging)
                found += 1
            else:
                layer.visible = False
    if found < 2:
        print "WARNING! Could not find appropriate artery/vein layers."


    print "Saved. "


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
