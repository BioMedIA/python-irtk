#!/usr/bin/python

import SimpleITK as sitk
import numpy as np
import cv2

import sys

shape = map( int, sys.argv[1:4] )
z = int(sys.argv[4])
y = int(sys.argv[5])
x = int(sys.argv[6])

f = sys.argv[7]
sitk_img = sitk.ReadImage( f )
data = sitk.GetArrayFromImage( sitk_img )

# ref = sys.argv[9]
# sitk_ref = sitk.ReadImage( f )

new_data = np.zeros( shape, dtype='int32' )
new_data[z:z+data.shape[0],
         y:y+data.shape[1],
         x:x+data.shape[2]] = data
new_img = sitk.GetImageFromArray( new_data )
new_img.SetDirection( sitk_img.GetDirection() )
new_img.SetOrigin( sitk_img.GetOrigin() )
new_img.SetSpacing( sitk_img.GetSpacing() )
sitk.WriteImage( new_img, sys.argv[8] ) 
