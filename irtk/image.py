from __future__ import division

__all__ = [ "Image",
            "imread",
            "imwrite",
            "imshow",
#            "viewer",
            "zeros",
            "ones",
            "new_header",
            "PointsToImage",
            "drawSphere",
            "sphere",
            "fix_orientation",
            "crf",
            "graphcut",
            "largest_connected_component",
            "landmarks_to_spheres" ]

import numpy as np
import pprint
import copy
import subprocess
import os

import scipy.ndimage as nd
from scipy import misc
from skimage import morphology

import _irtk
import registration
import utils

import IPython.display
#from IPython.html.widgets import interact
#from IPython.html import widgets
from io import BytesIO
from PIL.Image import fromarray


def new_header( pixelSize=[1,1,1,1],
                orientation=None,
                origin=[0,0,0,0],
                dim=None ):
    """
    Create new header.
    """
    if orientation is None:
        orientation = np.eye( 3, dtype='float64' )
    if dim is None:
        raise ValueError( "at least dim must be specified" )
    return { 'pixelSize' : np.array( pixelSize, dtype='float64'),
             'orientation' : orientation,
             'origin' : np.array( origin, dtype='float64'),
             'dim' : np.array( dim, dtype='int32') }

class Image(np.ndarray):
    """
    A subclass of numpy.ndarray holding the information from the NIFTI headers.
    """
    def __new__( cls,
                 img=None,
                 header=None,
                 copy=True,
                 squeeze=True,
                 force_neurological=False ):
        """
        constructor
        """
        if img is None:
            img = np.zeros(1)
        # else:
        #     img = np.array( img, copy=copy, order='C' )
            
        if header is None:
            dim = img.shape
            dim = list(dim)[::-1] + [1]*(4-len(dim))
            header = {
                'pixelSize' : np.ones(4,dtype='float64'),
                'orientation' : np.eye(3,dtype='float64'),                   
                'origin' : np.zeros(4,dtype='float64'),
                'dim' : np.array(dim,dtype='int32')
                }

        if squeeze:
            img = np.squeeze(img)
        # else:
        #     if len(img.shape) == 2:
        #         img = img[np.newaxis,np.newaxis,...]
        #     elif len(img.shape) == 3:
        #         img = img[np.newaxis,...]
                
        obj = np.asarray(img).view(cls)

        for i in xrange(4):
            if header['dim'][i] == 0:
                header['dim'][i] = 1
        
        obj.header = header
        obj.I2W = np.eye(4,dtype='float64')
        obj.W2I = np.eye(4,dtype='float64')

        obj.__updateMatrix()

        # https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;90445f5f.0710
        if force_neurological and np.linalg.det(obj.I2W[:3,:3]) < 0:
            obj.header['orientation'][0] *= -1.0
            img = img[:,:,::-1]
            return Image(img,header)
        
        return obj

    def __repr__(self):
        text = "Image data:\n" + repr(self.view(np.ndarray)) + "\n\n"
        text += "Image shape: " + repr(self.shape) + "\n\n"
        if self.header is not None:
            text += "Header:\n" + pprint.pformat(self.header,indent=4)
        else:
            text += "No header information\n"
        return text

    def __str__(self):
        if len(self.shape) == 0:
            return str(self.view(np.ndarray))
        else:
            return self.__repr__()

    def __array_finalize__(self, parent):
        # print 'In __array_finalize__:'
        # print self.shape, parent.shape
        # print '   self is %s' % repr(self)
        # print '   obj is %s' % repr(obj)
        if parent is None: return
        if self.shape != parent.shape:
            self.header = None
            self.I2W = None
            self.W2I = None
        else:
            self.header = getattr(parent, 'header', None)
            self.I2W = getattr(parent, 'I2W', None)
            self.W2I = getattr(parent, 'W2I', None)

    # def __array_wrap__(self, out_arr, context=None):
    #     print 'In __array_wrap__:'
    #     print self.shape, out_arr.shape
    #     # print '   self is %s' % repr(self)
    #     # print '   arr is %s' % repr(out_arr)
    #     #print context
    #     # then just call the parent
    #     return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, index):
        # TODO: time shift
        #print 'In __getitem__:', index
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple): index = (index,)
        # for i in index:
        #     print type(i),i 
            
        shift = []
        for i,slice_ in enumerate(index):
            if slice_ is Ellipsis:
                raise "use of Ellipsis not implemented"
            elif isinstance(slice_, slice):
                if slice_.start >= 0:
                    shift.append((i,slice_.start,slice_.stop))
            elif isinstance(slice_, (int, long)):
                shift.append((i,slice_,slice_+1))
            else:
                # result is no longer an Image
                # raise "wrong index:", index
                return np.ndarray.__getitem__(self.view(np.ndarray), index)
                
        header = self.get_header()
        mapping = []
        axis_names = ['x','y','z','t']
        axis = {'x':0,'y':1,'z':2,'t':3}
        for i in xrange(4):
            if (header['dim'][i] > 1) or (len(self.shape) == 4):
                mapping.append(axis_names[i])

        mapping = mapping[::-1]
        o1 = np.zeros(3,dtype='float64')
        for i, start, stop in shift:

            if stop < start:
                raise "negative slices not allowed"
            header['dim'][axis[mapping[i]]] = stop - start
            if mapping[i] == 'x':
                o1[0] = float(start)
            if mapping[i] == 'y':
                o1[1] = float(start)
            if mapping[i] == 'z':
                o1[2] = float(start)
        
        data = np.ndarray.__getitem__(self.view(np.ndarray), index)#.copy('C')

        # Calculate position of first voxel in roi in original image
        o1 = self.ImageToWorld( o1 )

        # Calculate position of first voxel in roi in new image
        tmp_img = Image(header=header)
        o2 = tmp_img.ImageToWorld( np.zeros(3) )

        # Shift origin of new image accordingly
        header['origin'][:3] += o1 - o2
  
        return Image( data, header, copy=False )

    def __reduce__(self):
        """
        Required for pickling/unpickling, which is used for instance
        in joblib Parallel.
        An example implementation can be found in numpy/ma/core.py .
        """        
        return ( _ImageReconstruct,
                 (self.get_data(), self.get_header()) )

    def __updateMatrix(self):
        #  Update image to world coordinate system matrix
        self.I2W = np.eye(4,dtype='float64')
        
        xAxis,yAxis,zAxis = self.header['orientation']
        self.I2W[0:3,0] = xAxis
        self.I2W[0:3,1] = yAxis
        self.I2W[0:3,2] = zAxis

        x,y,z,t = map(float,self.header['dim'])
        tmp1 = np.eye(4,dtype='float64')
        tmp1[0:3,3] = [-(x-1)/2,-(y-1)/2,-(z-1)/2]

        dx,dy,dz,dt = self.header['pixelSize']
        tmp2 = np.eye(4,dtype='float64')*[dx,dy,dz,1]

        tmp3 = np.eye(4,dtype='float64')
        tmp3[0:3,3] = self.origin()

        self.I2W = np.dot(tmp3, np.dot(self.I2W, np.dot(tmp2, tmp1)))

        # Update world to image coordinate system matrix
        self.W2I = np.eye(4,dtype='float64')

        self.W2I[0,0:3] = xAxis
        self.W2I[1,0:3] = yAxis
        self.W2I[2,0:3] = zAxis

        tmp1 = np.eye(4,dtype='float64')
        tmp1[0:3,3] = [(x-1)/2,(y-1)/2,(z-1)/2]

        tmp2 = np.eye(4,dtype='float64')*[1/dx,1/dy,1/dz,1]

        tmp3 = np.eye(4,dtype='float64')
        tmp3[0:3,3] = - self.origin()

        self.W2I = np.dot(tmp1, np.dot(tmp2, np.dot(self.W2I, tmp3)))

    def origin(self):
        """
        Return origin as a 3D point.
        """
        return self.header['origin'][:3]

    def nb_pixels(self,world=False):
        """
        Return number of pixels.
        """
        dim = self.header['dim']
        size = dim[0]*dim[1]*dim[2]*dim[3]
        if not world:
            return size
        else:
            ds = self.header['pixelSize']
            return size*ds[0]*ds[1]*ds[2]
    
    def ImageToWorld( self, pt ):
        """
        BEWARE: images are indexed as img[t,z,y,x],
        so for instance:

        Examples
        --------
        >>> tx,ty,tz = img.ImageToWorld( [(img.shape[2]-1)/2,(img.shape[1]-1)/2,(img.shape[0]-1)/2] )
        """
        # make sure I2W an W2I are up-to-date
        self.__updateMatrix()
        tmp_pt = np.array( pt, dtype='float64' ).copy()
        if len(tmp_pt.shape) == 1:
            tmp_pt = np.hstack((tmp_pt,[1])).astype('float64')
            return np.dot( self.I2W, tmp_pt )[:3]
        else:
            tmp_pt = _irtk.transform_points( self.I2W, tmp_pt )
            return tmp_pt            
            #tmp_pt = np.hstack((pt,[[1]]*pt.shape[0])).astype('float64')
            #return np.transpose( np.dot( self.I2W, np.transpose(tmp_pt) ) )[:,:3]
            #return np.array( map( self.ImageToWorld, pt) )
            # return np.dot( self.I2W, tmp_pt )[:,:3]
    
    def WorldToImage( self, pt ):
        """
        Returns integers
        """
        # make sure I2W an W2I are up-to-date
        self.__updateMatrix()
        tmp_pt = np.array( pt, dtype='float64' ).copy()
        if len(tmp_pt.shape) == 1:
            tmp_pt = np.hstack((tmp_pt,[1])).astype('float64')
            return np.rint( np.dot( self.W2I, tmp_pt )[:3] ).astype('int32')
        else:
            tmp_pt = np.rint( _irtk.transform_points( self.W2I, tmp_pt ) ).astype('int32')
            return tmp_pt
            #tmp_pt = np.hstack((pt,[[1]]*pt.shape[0])).astype('float64')
            #return np.rint( np.transpose( np.dot( self.W2I, np.transpose(tmp_pt) ) )[:,:3])
            #return np.array( map( self.WorldToImage, pt) )
                 
    def get_header(self):
        """
        Safe copy of the headers.
        """
        return copy.deepcopy(self.header)
    def set_header( self, new_header ):
        """
        Set the headers from a new dict.
        """
        self.header = copy.deepcopy(new_header)
        self.__updateMatrix()
    def get_data(self, dtype=None, purpose="python" ):
        """
        Return pixel data.
        """
        if dtype is None:
            dtype = self.dtype
        if purpose == "cython":
            shape = self.header['dim'][::-1]
            data = np.reshape( self.view(np.ndarray),
                               shape ).astype( dtype,
                                               copy=True,
                                               order='C')
        else:
            data = np.array( self, dtype=dtype, copy=True, order='C' )
            
        return data

    def copy( self, dtype=None ):
        """Returns a deep copy of the object."""
        return Image( self.get_data(dtype), self.get_header(), squeeze=False )

    def saturate( self, q0=1.0, q1=99.0, nonzero=False ):
        """
        Saturate pixel intensities using numpy.percentile.
        """
        data = self.get_data('float32')
        if q0 is None:
            q0 = 0.0
        if q1 is None:
            q1 = 100.0
        if nonzero:
            q = np.percentile(data[np.nonzero(data)],[q0,q1])
        else:
            q = np.percentile(data.flatten(),[q0,q1])
        data[data<q[0]] = q[0]
        data[data>q[1]] = q[1]
        return Image( data, self.header )
    
    def rescale( self, min=0, max=255, nonzero=False ):
        """ Stretch contrast."""
        data = self.get_data('float32')
        if nonzero:
            nonzero_indices = np.nonzero(data)
            data[nonzero_indices] -= data[nonzero_indices].min()
        else:
            data -= data.min()
        m = data.max()
        if m != 0:
            data /= m
        data = data * (max - min) + min
        return Image( data, self.header )

    def resample( self,
                  pixelSize=[1,1,1],
                  interpolation='linear',
                  gaussian_parameter=1.0,
                  labels=False,
                  label_smoothing=3.0 ):
        """
        Resample.
        """
        if isinstance(pixelSize,tuple):
            pixelSize = list(pixelSize)
        if not isinstance(pixelSize,(list,np.ndarray)):
            pixelSize = [pixelSize]*3
        pixelSize = np.asarray(pixelSize, dtype='float64')

        if not labels:
            new_img, new_header = _irtk.resample( self.get_data('float32', 'cython'),
                                                  self.get_header(),
                                                  pixelSize,
                                                  interpolation,
                                                  gaussian_parameter )
            return Image( new_img, new_header )
        else:
            nb_labels = self.max()+1
            tmp_stack = [] 
            for l in range(nb_labels):
                tmp_img = (self == l).astype('float32')
                tmp_stack.append( tmp_img.resample( pixelSize,
                                                    interpolation,
                                                    gaussian_parameter,
                                                    labels=False
                                                    ).gaussianBlurring(label_smoothing) )
            res = np.argmax( tmp_stack, axis=0)
            return Image( res, tmp_stack[0].get_header() )

    def resample2D( self,
                    pixelSize=[1,1],
                    interpolation='linear',
                    gaussian_parameter=1.0 ):
        """
        Resample 2D slices.
        """
        # FIXME: do this in C++
        if isinstance(pixelSize,tuple):
            pixelSize = list(pixelSize)
        if not isinstance(pixelSize, (list,np.ndarray)):
            pixelSize = [pixelSize]
        if len(pixelSize) == 1:
            pixelSize = pixelSize*2
        if len(pixelSize) > 2:
            raise ValueError( "this function is for XY resampling only" )
        
        header = self.get_header()
        # Determine the new dimensions of the image
        dim = header['dim']
        new_x = int(float(dim[0]) * header['pixelSize'][0] / pixelSize[0])
        new_y = int(float(dim[1]) * header['pixelSize'][1] / pixelSize[1])
        header['pixelSize'][0] = pixelSize[0]
        header['pixelSize'][1] = pixelSize[1]
        header['dim'][0] = new_x
        header['dim'][1] = new_y
        res = zeros( header, dtype='float32' )
        if len(res.shape) == 4:
            for t in xrange(self.shape[0]):
                for z in xrange(self.shape[1]):
                    res[t,z] = self[t,z].resample( header['pixelSize'],
                                                    interpolation=interpolation,
                                                    gaussian_parameter=gaussian_parameter)
        elif len(res.shape) == 3:
            for z in xrange(self.shape[0]):
                res[z] = self[z].resample( header['pixelSize'],
                                            interpolation=interpolation,
                                            gaussian_parameter=gaussian_parameter)
        else:
            res = self.resample( header['pixelSize'],
                                  interpolation=interpolation,
                                  gaussian_parameter=gaussian_parameter)
        return res

    def gaussianBlurring( self, sigma=1.0 ):
        """
        Gaussian blurring
        """
        new_img, new_header = _irtk.gaussianBlurring( self.get_data('float32', 'cython'),
                                                      self.get_header(),
                                                      sigma )
        return Image( new_img, new_header )

    def gdt( self, mask, l=10.0 ):
        """
        Gaussian blurring
        """
        new_img, new_header = _irtk.gdt( self.get_data('float32', 'cython'),
                                         mask.get_data('uint8', 'cython'),
                                         self.get_header(),
                                         l )
        return Image( new_img, new_header )

    def anisoDiff( self, iterations=5 ):
        """
        Anisotropic diffusion
        """
        new_img, new_header = _irtk.anisoDiff( self.get_data('float32', 'cython'),
                                               self.get_header(),
                                               iterations )
        return Image( new_img, new_header )    
    
    def register( self, target, transformation=None ):
        """
        Perform image registration.
        """
        return registration.registration_rigid( self, target, transformation )

    def transform( self,
                   transformation=None,
                   target=None,
                   interpolation='linear',
                   gaussian_parameter=1.0,
                   labels=False,
                   label_smoothing=3.0 ):
        """
        Transform image.
        """
        if transformation is None:
            transformation = registration.RigidTransformation()
        if not labels:
            return transformation.apply( self,
                                         target,
                                         interpolation,
                                         gaussian_parameter )
        else:
            nb_labels = self.max()+1
            tmp_stack = [] 
            for l in range(nb_labels):
                tmp_img = (self == l).astype('float32')
                tmp_img = tmp_img.gaussianBlurring( self.header[''])
                tmp_stack.append( tmp_img.transform( transformation,
                                                     target,
                                                     interpolation,
                                                     gaussian_parameter,
                                                     labels=False
                                                     ).gaussianBlurring(label_smoothing) )
            res = np.argmax( tmp_stack, axis=0)
            return Image( res, tmp_stack[0].get_header() )

    def mask( self, mask, threshold=0  ):
        """
        Mask image (after transforming the mask if necessary).
        """
        # transform mask to current image coordinate system
        tmp_mask = mask.transform( target=self )
        new_img = self.copy()
        new_img[ tmp_mask <= threshold ] = 0
        return new_img

    def reset_header( self, header=None ):
        """
        Reset header.
        """
        return Image( self.view(np.ndarray), header )

    def thumbnail( self,
                   seg=None,
                   overlay=None,
                   colors=None,
                   opacity=0.5,
                   step=2,
                   unroll=False,
                   index=None ):
        """
        Create a thumbnail.
        """
        img = self.copy('int')
        if overlay is not None and seg is not None:
            print "You cannot specify both seg and overlay"
            return
        if seg is not None:
            seg = seg.astype('int')
        if overlay is not None and colors is None:
            colors = 'jet' # default colormap
        if colors is not None and seg is None and overlay is None:
            overlay = img.copy() # we want to see img in colors
            img = zeros(img.get_header(), dtype='uint8')
            opacity = 1.0
        if isinstance( colors, str ):
            colors = utils.get_colormap( colors )
        if seg is not None and colors is None:
            colors = utils.random_colormap(seg.max())

        if index is None:
            index = np.array(img.shape).astype('int32')/2
            
        # now, seg == overlay
        if overlay is not None:
            seg = overlay

        if len(img.shape) == 2:
            if seg is not None:
                data = img.get_data('uint8')
                data = data.reshape(data.shape[0],
                                    data.shape[1],
                                    1)
                data = np.concatenate( [ data,
                                         data,
                                         data ], axis=2 )
                rgb_overlay = utils.remap( seg, colors )
                # op = (1-opacity) * (seg != 0).astype('float') + (seg == 0).astype('float')
                # op = op.reshape(op.shape[0],
                #                 op.shape[1],
                #                 1)
                # op = np.concatenate( [ op,
                #                        op,
                #                        op ], axis=2 )
                # img = op * data  + opacity*rgb_overlay
                img = (1-opacity) * data + opacity*rgb_overlay

            return img.astype('uint8')
        
        elif len(img.shape) == 3:
            if not unroll:
                shape = np.array(img.shape).max()
                if seg is None:
                    output = np.ones((shape,shape*3+step*(3-1)))*255
                else:
                    output = np.ones((shape,shape*3+step*(3-1),3))*255

                offset1 = int((shape - img.shape[1])/2)
                offset2 = int((shape - img.shape[2])/2)
                if seg is None:
                    tmp_img = img[index[0],:,:]
                else:
                    tmp_img = Image(img[index[0],:,:]).thumbnail( seg[index[0],:,:], colors=colors, opacity=opacity )
                output[offset1:offset1+img.shape[1],
                       offset2:offset2+img.shape[2]] = tmp_img

                offset1 = int((shape - img.shape[0])/2)
                offset2 = int(shape + step + (shape - img.shape[2])/2)
                if seg is None:
                    tmp_img = img[:,index[1],:]
                else:
                    tmp_img = Image(img[:,index[1],:]).thumbnail( seg[:,index[1],:], colors=colors, opacity=opacity)
                output[offset1:offset1+img.shape[0],
                       offset2:offset2+img.shape[2]] = tmp_img

                offset1 = int((shape - img.shape[0])/2)
                offset2 = int(2*shape + 2*step + (shape - img.shape[1])/2)
                if seg is None:
                    tmp_img = img[:,:,index[2]]
                else:
                    tmp_img = Image(img[:,:,index[2]]).thumbnail( seg[:,:,index[2]], colors=colors, opacity=opacity )
                output[offset1:offset1+img.shape[0],
                       offset2:offset2+img.shape[1]] = tmp_img

                return output.astype('uint8')
            else: # unroll is True
                if seg is None:
                    output = np.ones( ( self.shape[1],
                                        self.shape[2]*self.shape[0]+2*(self.shape[0]-1) )
                                      ) * 255
                else:
                    output = np.ones( ( self.shape[1],
                                        self.shape[2]*self.shape[0]+2*(self.shape[0]-1),
                                        3 )
                                      ) * 255
                for k in xrange(self.shape[0]):
                    if seg is None:
                        tmp_img = img[k,:,:]
                    else:
                        tmp_img = Image(img[k,:,:]).thumbnail( seg[k,:,:], colors=colors, opacity=opacity )
                    output[:,
                        k*self.shape[2]+2*k:(k+1)*self.shape[2]+2*k] = tmp_img
                return output.astype('uint8')
        else:
            raise "Wrong number of dimensions for thumbnail: " + str(len(self.shape)) 

    # We need to override certain functions so that they do not return an image
    # https://github.com/numpy/numpy/blob/master/numpy/matrixlib/defmatrix.py
    # def max(self, **kwargs):
    #     return self.view(np.ndarray).max(**kwargs)
    # def min(self, **kwargs):
    #     return self.view(np.ndarray).min(**kwargs)
    # def sum(self, **kwargs):
    #     return self.view(np.ndarray).sum(**kwargs)

    # def unroll( self ):
    #     output = np.ones( ( self.shape[1],
    #                         self.shape[2]*self.shape[0]+2*(self.shape[0]-1) )
    #                       ) * 255
    #     for k in xrange(self.shape[0]):
    #         output[:,
    #             k*self.shape[2]+2*k:(k+1)*self.shape[2]+2*k] = self[k,:,:]

    #     return output

    def wbox(self):
        tmp = self.bbox(crop=True)
        pts = np.array([[0,0,0],
                        [0,0,tmp.header['dim'][2]],
                        [0,tmp.header['dim'][1],0],
                        [tmp.header['dim'][0],0,0],
                        [0,tmp.header['dim'][1],tmp.header['dim'][2]],
                        [tmp.header['dim'][0],0,tmp.header['dim'][2]],
                        [tmp.header['dim'][0],tmp.header['dim'][1],0],
                        [tmp.header['dim'][0],tmp.header['dim'][1],tmp.header['dim'][2]]])
        pts = tmp.ImageToWorld( pts )
        x_min, y_min, z_min = pts.min(axis=0)
        x_max, y_max, z_max = pts.max(axis=0)
        return (x_min, y_min, z_min, x_max, y_max, z_max)
        
    def bbox( self, crop=False, world=False, lcc=False ):
        """
        Bounding box.
        """
        if lcc:
            mask = largest_connected_component(self)
        else:
            mask = self > 0
        pts = np.transpose(np.nonzero(mask))
        if world:
            pts = self.ImageToWorld( pts[:,::-1] )
            x_min, y_min, z_min = pts.min(axis=0)
            x_max, y_max, z_max = pts.max(axis=0)
            return (x_min, y_min, z_min, x_max, y_max, z_max)
        if len(self.shape) == 2:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            if not crop:
                return (x_min, y_min, x_max, y_max )
            else:
                return self[y_min:y_max+1,
                            x_min:x_max+1]        
        if len(self.shape) == 3:
            z_min, y_min, x_min = pts.min(axis=0)
            z_max, y_max, x_max = pts.max(axis=0)
            if not crop:
                return (x_min, y_min, z_min, x_max, y_max, z_max )
            else:
                return self[z_min:z_max+1,
                            y_min:y_max+1,
                            x_min:x_max+1]
        if len(self.shape) == 4:
            t_min, z_min, y_min, x_min = pts.min(axis=0)
            t_max, z_max, y_max, x_max = pts.max(axis=0)
            if not crop:
                return (x_min, y_min, z_min, t_min, x_max, y_max, z_max, t_max )
            else:
                return self[t_min:t_max+1,
                            z_min:z_max+1,
                            y_min:y_max+1,
                            x_min:x_max+1]            

    def order( self ):
        """
        Return either "radiological" or "neurological".
        """
        if np.linalg.det(self.I2W[:3,:3]) < 0:
            return "radiological"
        else:
            return "neurological"
    def neurological( self ):
        """
        Return True if neurological order.
        """
        return self.order() == "neurological"
    def radiological( self ):
        """
        Return True if radiological order.
        """
        return self.order() == "radiological"
    def orientation( self ):
        """
        Return orientation.
        """
        return _irtk.orientation( self.get_header() )

    def dice( self, mask2, verbose=False ):
        """
        Dice and overlap metrics
        """
        countA = self.sum()
        countB = mask2.sum()
        countAandB = np.sum( np.logical_and(self, mask2) )

        fraction = (countA+countB)/2
        if fraction != 0:
            dice_metric = float(countAandB)/float(fraction) 
        else:
            dice_metric = 0

        fraction = countA+countB-countAandB
        if fraction != 0:
            overlap_metric = float(countAandB)/float(fraction)
        else:
            overlap_metric = 0

        if verbose:
            print "Dice metric: ", dice_metric
            print "Overlap metric: ", overlap_metric

        return dice_metric, overlap_metric

    def resize( self, shape, interpolation='linear' ):
        """
        Resize image with an affine transformation to force it to a given shape.
        """
        dx,dy,dz,dt = self.header['pixelSize']
        dx *= self.shape[2] / shape[2]
        dy *= self.shape[1] / shape[1]
        dz *= self.shape[0] / shape[0]

        target = self.get_header()
        target['pixelSize'] = np.array( [dx,dy,dz,dt], dtype='float64' )
        target['dim'] = np.array( [shape[2],shape[1],shape[0],1], dtype='int32' )

        return self.transform( target=target, interpolation=interpolation )

    def as3D( self ):
        """Force 3D shape"""
        if len(self.shape) == 2:
            self = Image( self[np.newaxis,...].copy(),
                          self.get_header(),
                          squeeze=False )
        return self

    def as4D( self ):
        """Force 4D shape"""
        if len(self.shape) == 2:
                self = Image( self[np.newaxis,np.newaxis,...].copy(),
                              self.get_header(),
                              squeeze=False )
        elif len(self.shape) == 3:
                self = Image( self[np.newaxis,...].copy(),
                              self.get_header(),
                              squeeze=False )
        return self
                
def imread( filename,
            dtype=None,
            empty=False,
            force_neurological=False ):
    """
    Reads an image in a format understandable by IRTK.

    Parameters
    ----------
    filename : str
        Name of the file to read.
    dtype : str, optional (default: None)
        Pixel type of the output image. If dtype is None, then the pixel type
        that has been used to write the file to disk is selected. Note that for the
        slope/intercept information to be taken into account in NIFTI files.
        If an integer type is requested and slope is not 1 or intercept not 0,
        then the following message will be printed on stderr:

        ``irtkGenericImage<float>::Read: Ignore slope and intercept, use irtkGenericImage<float> or
        irtkGenericImage<double> instead``

    force_neurological : bool, optional (default: False)
        Forces the image to be read with the neurological convention. See the
        notes on neurological/radiological convention for more details.

        
    Returns
    -------
    img : irtk.Image
        A subclass of numpy.ndarray holding the information from the NIFTI headers.

    Notes
    -----

    Note on neurological/radiological order:

    In neurological order, the determinant of the ImageToWorld matrix has a positive
    determinant: the image is thus represented in a direct orthonormal system, you
    are positioning yourself in the shoes of the patient, the right of the patient
    is at the right of the image. 

    In radiological order, the determinant of the ImageToWorld matrix has a negative
    determinant: the image is thus represented in a indirect orthonormal system, you
    are looking at a patient which is looking at you, as in a mirror, the right of
    the patient is at the left of the image. 

    To go from radiological to neurological, you need to mirror the pixel data along
    the X axis as well as multiply the X axis by -1 in the header information. It is
    only the combination of the two operations that ensure that the information
    contained in the image is still the same. This is valid for IRTK which places
    the image origin at the center of the image, but an additional offset should be
    taken into account if the origin of the image is defined differently (for
    instance position [0,0,0] in the array of pixels).
        
    Examples
    --------
    >>> img = irtk.imread( "input.nii", dtype="float32" )
    >>> print img[0:2,0:2,0:2]
    Image data:
    array([[[ 455.95440674,  452.3605957 ],
            [ 424.96868896,  419.69546509]],
    <BLANKLINE>
           [[ 462.80618286,  477.69018555],
            [ 433.81433105,  455.43170166]]], dtype=float32)
    <BLANKLINE>
    Image shape: (2, 2, 2)
    <BLANKLINE>
    Header:
    {   'dim': array([2, 2, 2, 1], dtype=int32),
        'orientation': array([[ 0.18479464, -0.98071011,  0.06370761],
                              [ 0.34170858,  0.12489687,  0.93146984],
                              [ 0.92145872,  0.15036121, -0.35819733]]),
        'origin': array([  6.61570985,  35.65394667, -74.69886408,   0.        ]),
        'pixelSize': array([ 0.75,  0.75,  0.75,  1.  ])}
    """
    name, extension = os.path.splitext(filename)
    if extension in [".jpg",".png"]:
        img = misc.imread(filename)
        if dtype is not None:
            img = img.astype(dtype)
        return Image(img)
    
    if dtype is "int": dtype = "int32"
    if dtype is "float": dtype = "float64"
    reader = {
        "int8" : _irtk.imread_char,
        "uint8" : _irtk.imread_uchar,
        "int16" : _irtk.imread_short,
        "uint16" : _irtk.imread_ushort,
        "int32" : _irtk.imread_int,
        "uint32" : _irtk.imread_uint,
        "float32" : _irtk.imread_float,
        "float64" : _irtk.imread_double,
        }

    header, original_dtype = _irtk.get_header( filename )
    if header is None:
        return None
    if empty:
        return Image(np.empty(0),header,force_neurological=force_neurological)
    if dtype is None:
        dtype = original_dtype
    img = reader[dtype]( filename, header )
    if img is False:
        return None
    else:
        return Image(img,header,force_neurological=force_neurological)

def imwrite( filename, img ):
    """
    Write image.
    """
    name, extension = os.path.splitext(filename)
    if extension in [".jpg",".png"]:
        misc.imsave( filename, np.squeeze( img ) )
        return True
    
    if not isinstance(img,Image):
        img = Image(img)
    if img.dtype.name == "bool":
        img = img.astype("uint8")
    if img.dtype.name == "int64":
        img = img.astype("int32")        
    writer = {
        "int8" : _irtk.imwrite_char,
        "uint8" : _irtk.imwrite_uchar,
        "int16" : _irtk.imwrite_short,
        "uint16" : _irtk.imwrite_ushort,
        "int32" : _irtk.imwrite_int,
        "uint32" : _irtk.imwrite_uint,
        "float32" : _irtk.imwrite_float,
        "float64" : _irtk.imwrite_double,
        }

    shape = img.header['dim'][::-1]
    data = np.reshape( img.view(np.ndarray), shape ).copy('C')
    success = writer[img.dtype.name]( filename, data, img.header )
    return success

def zeros( header, dtype='float32' ):
    """
    Similar to numpy.zeros, but with given headers.
    """
    return Image( np.zeros( header['dim'][::-1], dtype=dtype ), header )

def ones( header, dtype='float32' ):
    """
    Similar to numpy.ones, but with given headers.
    """
    return Image( np.ones( header['dim'][::-1], dtype=dtype ), header ) 

def write_list( img_list ):
    n = len(img_list)
    nb_pixels = 0
    for img in img_list:
        nb_pixels += img.nb_pixels()

    pixelData = np.zeros( nb_pixels, dtype='float32' )
    pixelSize = np.zeros( 4*n, dtype='float64' )
    xAxis = np.zeros( 3*n, dtype='float64' )
    yAxis = np.zeros( 3*n, dtype='float64' )
    zAxis = np.zeros( 3*n, dtype='float64' )
    origin = np.zeros( 4*n, dtype='float64' )
    dim = np.zeros( 4*n, dtype='int32' )
    
    offset = 0
    for i, img in enumerate(img_list):
        pixelData[offset:offset+img.nb_pixels()] = img.flatten()
        offset += img.nb_pixels()
        pixelSize[4*i:4*(i+1)] = img.header['pixelSize']
        xAxis[3*i:3*(i+1)] = img.header['orientation'][0]
        yAxis[3*i:3*(i+1)] = img.header['orientation'][1]
        zAxis[3*i:3*(i+1)] = img.header['orientation'][2]
        origin[4*i:4*(i+1)] = img.header['origin']
        dim[4*i:4*(i+1)] = img.header['dim']

    print pixelData.shape

    _irtk.write_list( pixelData, pixelSize, xAxis, yAxis, zAxis, origin, dim, n )

def PointsToImage( pts, header ):
    """
    Create an image from a set of 3D points.
    """
    if isinstance(header,Image):
        header = header.get_header()
    pts = np.array(pts, dtype="float64")
    img = _irtk.points_to_image( pts, header )
    return Image( img, header )

def drawSphere( img, (x,y,z), rx, ry=None, rz=None ):
    """
    Draw a sphere in a given image.
    """
    if ry is None:
        ry = rx
    if rz is None:
        rz = rx
    _irtk.drawSphere( img, x, y, z, rx, ry, rz )

def sphere( (x, y, z), r, header ):
    """
    Create an image with a filled sphere.
    """
    img = zeros( header, dtype='uint8' )
    x0, y0, z0 = img.WorldToImage( (x, y, z) )
    rx = int(round( r / header['pixelSize'][0] ))
    ry = int(round( r / header['pixelSize'][1] ))
    rz = int(round( r / header['pixelSize'][2] ))
    print "Sphere:",  x0, y0, z0, rx, ry, rz
    _irtk.drawSphere( img, x0, y0, z0, rx, ry, rz )
    return img #Image( img, header )

def _ImageReconstruct( data, header ):
    """Internal function that builds a new Image from the
    information stored in a pickle.
    """
    return Image( data, header, squeeze=False )

def fix_orientation( img1, img2, verbose=False ):
    order1 = img1.order()
    order2 = img2.order()
    if verbose:
        print "Image1 is in " + order1 + " order"
        print "Image2 is in " + order2 + " order"
    if order1 == order2:
        if verbose:
            print "nothing to fix"
        return img2
    i1, j1, k1 = img1.orientation()
    i2, j2, k2 = img2.orientation()

    header = img2.get_header()
    data = img2.get_data()

    if i1 != i2:
        data = data[:,:,::-1]
        header['orientation'][0] *= -1
        if verbose:
            print "flipping X axis"

    if j1 != j2:
        data = data[:,::-1,:]
        header['orientation'][1] *= -1
        if verbose:
            print "flipping Y axis"

    if k1 != k2:
        data = data[::-1,:,:]
        header['orientation'][2] *= -1
        if verbose:
            print "flipping Z axis"

    return Image( data.copy(), header )

def rgb(img):
    new_img = np.zeros((3,img.shape[0],img.shape[1]),dtype="uint8")
    for i in xrange(3):
        new_img[i] = img[:,:,i]
    return new_img

def imshow( img,
            seg=None,
            overlay=None,
            colors=None,
            opacity=0.5,
            filename=None,
            unroll=False,
            index=None ):
    """
    Display a 2D image
    """
    if len(img.shape) not in [2,3]:
        print "Invalid image shape: ", img.shape
        return
    
    if filename is None:
        embed = True
    else:
        embed = False

    img = img.rescale()
    img = img.thumbnail( seg=seg,
                         overlay=overlay,
                         colors=colors,
                         opacity=opacity,
                         unroll=unroll,
                         index=index ).astype("uint8")

    #if len(img.shape) == 3 and img.shape[2] == 3:
    #    img = rgb(img)

    if embed == True:
        # http://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
        b = BytesIO()
        #fromarray( np.transpose(img.view(np.ndarray),
        #                        (1,2,0)).astype('uint8')).save(b, format='png')
        fromarray( img.view(np.ndarray) ).save(b, format='png')
        return IPython.display.Image( data=b.getvalue() )
    else:
        if imwrite(filename, img ):
            return IPython.display.Image( filename=filename,
                                               url='file://'+os.path.abspath(filename),
                                               embed=False )
        else:
            return False

#def viewer( img,
#            seg=None,
#            overlay=None,
#            colors=None,
#            opacity=0.5,
#            index=None ):
#    """Inline viewer for IPython Notebook""" 
#
#    def f(z,y,x):
#        IPython.display.display( imshow(img,seg,overlay,colors,opacity,index=(z,y,x)))
#
#    if index is None:
#        index = (np.array(img.shape)/2).astype('int')
#        
#    interact( f,
#              z=widgets.IntSliderWidget(min=0,max=img.shape[0]-1,step=1,value=index[0]),
#              y=widgets.IntSliderWidget(min=0,max=img.shape[1]-1,step=1,value=index[1]),
#              x=widgets.IntSliderWidget(min=0,max=img.shape[2]-1,step=1,value=index[2]))
    
def crf( img, labels, proba, l=1.0, sigma=0.0, sigmaZ=0.0 ):
    """
    Conditional Random Field.
    """
    if not isinstance(img,Image): img = Image(img)
    if not isinstance(labels,Image): labels = Image(labels)
    if not isinstance(proba,Image): proba = Image(proba)
    header = img.get_header()
    new_labels = _irtk.crf( img.get_data('float64','cython'),
                            header,
                            labels.get_data('int16','cython'),
                            proba.get_data('float64','cython'),
                            l, sigma, sigmaZ )
    return Image(new_labels, header)

def graphcut( img, labels, l=1.0, sigma=0.0, sigmaZ=0.0 ):
    """
    Graphcut.
    """
    if not isinstance(img,Image): img = Image(img)
    if not isinstance(labels,Image): labels = Image(labels)
    header = img.get_header()
    new_labels = _irtk.graphcut( img.get_data('float64','cython'),
                                 header,
                                 labels.get_data('int16','cython'),
                                 l, sigma, sigmaZ )
    return Image(new_labels, header)
    
def largest_connected_component( img, fill_holes=False ):
    """Largest connected component"""
    labels, nb_labels = nd.label(img>0)
    res = img.copy()
    if nb_labels > 1:
        sizes =  np.bincount( labels.flatten(),
                              minlength=nb_labels+1 )
        sizes[0] = 0
        i_max = np.argmax( sizes )
        res[labels!=i_max] = 0

    if fill_holes:
        res[nd.binary_fill_holes(res)>0] = 1
    
    return res
    
def landmarks_to_spheres(seg, r=10):
    """Landmarks to spheres"""
    nb_landmarks = np.unique(seg)[1:]
    centers = []
    for l in nb_landmarks:
        centers.append( nd.center_of_mass( (seg == l).view(np.ndarray) ) )
    centers= np.array(centers)[:,::-1]
    centers = seg.ImageToWorld(centers)

    res = zeros( seg.get_header(), dtype='uint8' )
    ball = Image( morphology.ball(r) )

    for c,l in zip(centers,nb_landmarks):
        ball.header['origin'] = c
        ball2 = ball.transform(target=res)
        res[ball2>0] = l
        
    return res
