#!/usr/bin/python
 
import irtk
from irtk.vtk2irtk import read_polydata, voxellise
from irtk.sitkreader import sitk_read
import sys
 
vtk_file = sys.argv[1]
img_file = sys.argv[2]
nifti_file = sys.argv[3]

points,triangles = read_polydata( vtk_file )

img = sitk_read( img_file )
print img.header, img.shape

img = voxellise(points,triangles,img.get_header())

irtk.imwrite(nifti_file,img)


