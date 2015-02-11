#!/usr/bin/python

import sys
import numpy as np

import irtk
import scipy.ndimage as nd

def get_noiseXY(img):
    img = img.astype('float32')
    new_img = np.zeros(img.shape,dtype='float32')
    for z in xrange(img.shape[0]):
        new_img[z] = nd.gaussian_filter( img[z], 2, mode='reflect' )
    noise = img - new_img
    #print "Noise XY:", noise.std(), img.std()
    return noise.std()

def get_noiseZ(img):
    img = img.astype('float32')
    new_img = np.zeros(img.shape,dtype='float32')
    for x in xrange(img.shape[2]):
        new_img[:,:,x] = nd.gaussian_filter( img[:,:,x], 2, mode='reflect' )
    noise = img - new_img
    #print "Noise Z:", noise.std(), img.std()
    return noise.std()

output_filename = sys.argv[3]

img = irtk.imread( sys.argv[1], dtype='float64' ).saturate()
mask = irtk.imread( sys.argv[2], dtype='int16' )
mask = irtk.Image(mask,img.get_header())

# crop
x_min,y_min,z_min,x_max,y_max,z_max = mask.bbox()
mask = mask[z_min:z_max+1,
            y_min:y_max+1,
            x_min:x_max+1]
tmp_img = img[z_min:z_max+1,
              y_min:y_max+1,
              x_min:x_max+1]


downsampled_img = tmp_img.resample(2)
mask = mask.transform(target=downsampled_img,interpolation='nearest')

seg = irtk.graphcut( downsampled_img, mask,
                     sigma=get_noiseXY(downsampled_img),
                     sigmaZ=get_noiseZ(downsampled_img) )

irtk.imwrite( sys.argv[3], seg.transform(target=img,interpolation='nearest') )


