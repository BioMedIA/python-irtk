#!/usr/bin/python

import SimpleITK as sitk
import numpy as np
import cv2

import sys

axis = int(sys.argv[1])

f = sys.argv[2]
sitk_img = sitk.ReadImage( f )
data = sitk.GetArrayFromImage( sitk_img )

if axis == 0:
    new_data = data[::-1,:,:].copy()
elif axis == 1:
    new_data = data[:,::-1,:].copy()
else:
    new_data = data[:,:,::-1].copy()
    
new_img = sitk.GetImageFromArray( new_data )
new_img.SetDirection( sitk_img.GetDirection() )
new_img.SetOrigin( sitk_img.GetOrigin() )
new_img.SetSpacing( sitk_img.GetSpacing() )
sitk.WriteImage( new_img, sys.argv[3] ) 
