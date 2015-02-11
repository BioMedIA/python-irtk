#!/usr/bin/python
 
import irtk
from irtk.sitkreader import sitk_read
import sys
 
input_file = sys.argv[1]
output_file = sys.argv[2]

img = sitk_read( input_file )
print img.header, img.shape

irtk.imwrite(output_file,img)


