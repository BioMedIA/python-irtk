#!/usr/bin/python

import SimpleITK as sitk
import numpy as np
import cv2

import sys

from scipy.stats.mstats import mquantiles
from skimage.feature import match_template

full_file = sys.argv[1]
cropped_file = sys.argv[2]

full_img = sitk.ReadImage( full_file )
full_data = sitk.GetArrayFromImage( full_img ).astype("float32")

cropped_img = sitk.ReadImage( cropped_file )
cropped_data = sitk.GetArrayFromImage( cropped_img ).astype("float32")

offset = cropped_data.shape[0]/2
cropped_data = cropped_data[offset]

res = []
for z in range(full_data.shape[0]):
    res.append( match_template( full_data[z], cropped_data, pad_input=False ) )

res = np.array(res)
print res.max()
z,y,x = np.unravel_index( np.argmax(res), res.shape )
z = z - offset

print z,y,x

print ' '.join( map(str, [full_data.shape[0],
                   full_data.shape[1],
                   full_data.shape[2],
        z,
        y,
        x]) )

