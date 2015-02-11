#!/usr/bin/python

import sys
import numpy as np
import irtk

from pyhull.convex_hull import ConvexHull
from irtk.vtk2irtk import voxellise

seg = irtk.imread(sys.argv[1])
output = sys.argv[2]

ZYX = np.transpose(np.nonzero(seg))
pts = seg.ImageToWorld( ZYX[:,::-1] )
hull = ConvexHull(pts)

img = voxellise(  hull.points, hull.vertices, header=seg.get_header() )

irtk.imwrite( output, img )
