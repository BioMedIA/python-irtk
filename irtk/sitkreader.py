from image import Image
import SimpleITK as sitk
import numpy as np

def sitk_read( filename ):
    img = sitk.ReadImage( filename )
    spacing = np.array( list(img.GetSpacing())+[1], dtype='float64' )
    img = Image( sitk.GetArrayFromImage(img) )
    img.header['pixelSize'] = spacing
    img.header['origin'][:3] += img.header['dim'][:3].astype('float')*img.header['pixelSize'][:3]/2
    return img
