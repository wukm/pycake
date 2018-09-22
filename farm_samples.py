#/usr/bin/env python

"""
This should be a plugin to take images from the folder NCS_vessel_GIMP_xcf
and create trace, mask, and backgrounded images from each xcf file.

What follows is actually without plugin syntax. The following commands are what
you would type directly into the python console in gimp.

"""
from gimpfu import *
import os.path

# can't use this because we can't run this outside of gimp i think?
# need to write a batch file to run this and then run # gimp -b ...

#for i, xcffile in enumerate(glob('*.xcf')):

#basefile, ext = os.path.splitext(xcffile)

# get active image
img = gimp.image_list()[0]

# Go to perimeter layer.
perimeter = pdb.gimp_image_get_layer_by_name(img, 'perimeter')
# could also iterate and say if layer.name = 'perimeter' ...

# Copy perimeter layer & focus new layer (only visible).

# .copy() has optional arg of "add_alpha_channel"
M = perimeter.copy()

# set all other layers non visible
for layer in img.layers:
    layer.visible = False

# add in position 0 (top)
img.add_layer(M, 0)

# Remove Alpha Channel.
pdb.gimp_layer_flatten(M)
## Invert Colors
#pdb.gimp_invert(M)


# color exchange yellow & blue to black
pdb.plug_in_exchange(img,m,255,255,0,0,0,0,1,1,1)
pdb.plug_in_exchange(img,m,0,0,255,0,0,0,1,1,1)

# set FG color to black
gimp.set_foreground(0,0,0)

cx, cy = img.height // 2 , img.width // 2

# Bucket Fill Inside black (middle x,y is probably OK)
pdb.gimp_edit_bucket_fill(m,0,0,100,0,0,cx,cy)
# Color Exchange Green to White.
pdb.plug_in_exchange(img,m,0,255,0,255,255,255,1,1,1)

# Color Exchange Cyan (00ffff) to White.
pdb.plug_in_exchange(img,m,0,255,255,255,255,255,1,1,1)

# Export Layer as Image called "f".mask.png

pdb.gimp_file_save(img,m, '/home/luke/test.png', '')

# invert (so back exterior now)
pdb.invert(m)
m.mode = DARKEN_ONLY_MODE # the constant 9

# set bottom layer (placenta) to visible
img.layers[-1].visible = True

# now make a new layer called 'main' from visible
raw_img = pdb.gimp_layer_new_from_visible(img,img,'raw_img')
img.add_layer(main,0)
pdb.gimp_file_save(img ,raw_img, '/home/luke/test2.png', '')

# now set the veins/artery layers as only visible
for layer in img.layers:
    layer.visible = (layer.name in ("arteries", "veins"))

trace = pdb.gimp_layer_new_from_visible(img,img,'trace')
img.add_layer(trace,0)

pdb.gimp_layer_flatten(trace) # remove alpha channel

pdb.gimp_desaturate(trace) # turn to grayscale
pdb.gimp_threshold(trace,255,255) # anything not 255 turns black

pdb.gimp_file_save(img, trace, '/home/luke/testtrace.png','')
# Visible to New Layer called Trace & focus (only visible)
# Remove Alpha Channel of layer.
# Threshold (127 is default and  OK)
# Invert Colors
# Export Layer as "f".trace.png
# Make "Background" Layer and Mask Layer Visible
# Move Mask Layer in Front of "Background" Layer
# Change Mask Layer Mode to "Darken Only"
# Export Visible as *.png
