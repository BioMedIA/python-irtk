"""
You want to match the source to the target.

"""

from __future__ import division

__all__ = [ "RigidTransformation",
            "AffineTransformation",
            "read_points",
            "registration_rigid_points",
            "registration_affine_points",
            "flirt2dof",
            "flirt_registration" ]

import numpy as np
from math import cos, sin, pi, asin, atan2
import os
import tempfile

import _irtk
import irtk
import utils

class RigidTransformation:
    """
    Rigid transformation class.
    """
    def __init__( self, filename=None, matrix=None,
                  tx=0, ty=0, tz=0,
                  rx=0, ry=0, rz=0 ):
        if filename is not None:
            ( tx, ty, tz,
              rx, ry, rz ) = _irtk.read_rigid( filename )
        elif matrix is not None:
            ( tx, ty, tz,
              rx, ry, rz ) = self.__from_matrix( matrix )
            
        ( self.tx, self.ty, self.tz,
          self.rx, self.ry, self.rz ) = map( float,
                                             ( tx, ty, tz,
                                               rx, ry, rz ) )

    def get_parameters( self ):
        """
        Return parameters.
        """
        return ( self.tx, self.ty, self.tz,
                 self.rx, self.ry, self.rz )

    def __repr__( self ):
        txt = "tx: " + str(self.tx) + " (mm)\n"
        txt += "ty: " + str(self.ty) + " (mm)\n"
        txt += "tz: " + str(self.tz) + " (mm)\n"
        txt += "rx: " + str(self.rx) + "(degrees)\n"
        txt += "ry: " + str(self.ry) + "(degrees)\n"
        txt += "rz: " + str(self.rz)+ " (degrees)"
        return txt

    def matrix( self ):
        """
        Return matrix.
        """
        cosrx = cos(self.rx*(pi/180.0))
        cosry = cos(self.ry*(pi/180.0))
        cosrz = cos(self.rz*(pi/180.0))
        sinrx = sin(self.rx*(pi/180.0))
        sinry = sin(self.ry*(pi/180.0))
        sinrz = sin(self.rz*(pi/180.0))

        # Create a transformation whose transformation matrix is an identity matrix
        m = np.eye( 4, dtype='float64' )

         # Add other transformation parameters to transformation matrix
        m[0,0] = cosry*cosrz
        m[0,1] = cosry*sinrz
        m[0,2] = -sinry
        m[0,3] = self.tx

        m[1,0] = (sinrx*sinry*cosrz-cosrx*sinrz)
        m[1,1] = (sinrx*sinry*sinrz+cosrx*cosrz)
        m[1,2] = sinrx*cosry
        m[1,3] = self.ty

        m[2,0] = (cosrx*sinry*cosrz+sinrx*sinrz)
        m[2,1] = (cosrx*sinry*sinrz-sinrx*cosrz)
        m[2,2] = cosrx*cosry
        m[2,3] = self.tz
        m[3,3] = 1.0

        return m

    def __from_matrix( self, m ):
        m = m.astype('float64')
        TOL = 0.000001

        tx = m[0,3]
        ty = m[1,3]
        tz = m[2,3]

        tmp = asin( -1.0 * m[0,2] )

        # asin returns values for tmp in range -pi/2 to +pi/2, i.e. cos(tmp) >=
        # 0 so the division by cos(tmp) in the first part of the if clause was
        # not needed.
        if abs(cos(tmp)) > TOL:
            rx = atan2(m[1,2], m[2,2])
            ry = tmp
            rz = atan2(m[0,1], m[0,0])
        else:
            # m[0,2] is close to +1 or -1
           rx = atan2(-1.0*m[0,2]*m[1,0], -1.0*m[0,2]*m[2,0])
           ry = tmp
           rz = 0
           
        # Convert to degrees.
        rx *= 180.0/pi
        ry *= 180.0/pi
        rz *= 180.0/pi

        return ( tx, ty, tz,
                 rx, ry, rz )

    def write( self, filename ):
        """
        Save to disk.
        """
        _irtk.write_rigid( filename,
                           self.tx, self.ty, self.tz,
                           self.rx, self.ry, self.rz )

    def apply( self, img, target_header=None,
               interpolation='linear', gaussian_parameter=1.0 ):
        """
        Apply.
        """
        if not isinstance( img, irtk.Image ):
            # point transformation
            pt = np.array(img, dtype='float64', copy=True)
            if len(pt.shape) == 1:
                tmp_pt = np.hstack((pt,[1])).astype('float64')
                return np.dot( self.matrix(), tmp_pt )[:3]
            else:
                pt = _irtk.transform_points( self.matrix(), pt )
                return pt  
                # tmp_pt = np.hstack((pt,[[1]]*pt.shape[0])).astype('float64')
                # return np.transpose( np.dot( self.matrix(),
                #                              np.transpose(tmp_pt) ) )[:,:3]
        
        # if target_header is None:
        #     target_header = img.get_header()
        if target_header is None:
            (x_min, y_min, z_min, x_max, y_max, z_max ) = img.bbox(world=True)
            corners = [[x_min, y_min, z_min],
                       [x_max, y_min, z_min],
                       [x_min, y_max, z_min],
                       [x_min, y_min, z_max],
                       [x_max, y_max, z_min],
                       [x_min, y_max, z_max],
                       [x_max, y_min, z_max],
                       [x_max, y_max, z_max]]
            corners = self.apply( corners )
            x_min, y_min, z_min = corners.min(axis=0)
            x_max, y_max, z_max = corners.max(axis=0)
            res = img.header['pixelSize'][0]
            pixelSize = [res, res, res, 1]
            origin = [ x_min + (x_max+1 - x_min)/2,
                       y_min + (y_max+1 - y_min)/2,
                       z_min + (z_max+1 - z_min)/2,
                       img.header['origin'][3] ]
            dim = [ (x_max+1 - x_min)/res,
                    (y_max+1 - y_min)/res,
                    (z_max+1 - z_min)/res,
                    1 ]
            target_header = irtk.new_header( pixelSize=pixelSize, origin=origin, dim=dim)
        if isinstance( target_header, irtk.Image ):
            target_header = target_header.get_header()
        data = img.get_data('float32','cython')
        new_data = _irtk.transform_rigid( self.tx, self.ty, self.tz,
                                          self.rx, self.ry, self.rz,
                                          data,
                                          img.get_header(),
                                          target_header,
                                          interpolation,
                                          gaussian_parameter )
        return irtk.Image( new_data, target_header )

    def invert( self ):
        """
        Invert.
        """
        return RigidTransformation( matrix=np.linalg.inv(self.matrix()) )

    def __mul__(self, other):
        """
        Overloading multiply operator to simulate the composition of transformations:
        returns self * other
        """
        return RigidTransformation( matrix=np.dot(self.matrix(),
                                             other.matrix()) )


def registration_rigid( source, target, transformation=None ):
    if transformation is None:
        transformation = RigidTransformation()
    tx, ty, tz, rx, ry, rz = transformation.get_parameters()
    tx, ty, tz, rx, ry, rz = _irtk.registration_rigid( source.get_data('int16',
                                                                       'cython'),
                                                       source.get_header(),
                                                       target.get_data('int16',
                                                                       'cython'),
                                                       target.get_header(),
                                                       tx, ty, tz,
                                                       rx, ry, rz )
    return RigidTransformation( tx=tx, ty=ty, tz=tz,
                                rx=rx, ry=ry, rz=rz )


class AffineTransformation:
    """
    Affine transformation class.
    """
    def __init__( self, filename=None, matrix=None,
                  tx=0, ty=0, tz=0,
                  rx=0, ry=0, rz=0,
                  sx=0, sy=0, sz=0,
                  sxy=0, syz=0, sxz=0 ):
        if filename is not None:
            ( tx, ty, tz,
              rx, ry, rz,
              sx, sy, sz,
              sxy, syz, sxz ) = _irtk.read_affine( filename )
        elif matrix is not None:
            ( tx, ty, tz,
              rx, ry, rz,
              sx, sy, sz,
              sxy, syz, sxz ) = self.__from_matrix( matrix )
            
        ( self.tx, self.ty, self.tz,
          self.rx, self.ry, self.rz,
          self.sx, self.sy, self.sz,
          self.sxy, self.syz, self.sxz ) = map( float,
                                                ( tx, ty, tz,
                                                  rx, ry, rz,
                                                  sx, sy, sz,
                                                  sxy, syz, sxz ) )

    def get_parameters( self ):
        """
        Return parameters.
        """
        return ( self.tx, self.ty, self.tz,
                 self.rx, self.ry, self.rz,
                 self.sx, self.sy, self.sz,
                 self.sxy, self.syz, self.xz )

    def to_rigid( self ):
        return RigidTransformation( tx=self.tx, ty=self.ty, tz=self.tz,
                                    rx=self.rx, ry=self.ry, rz=self.rz )

    def __repr__( self ):
        txt = "tx: " + str(self.tx) + " (mm)\n"
        txt += "ty: " + str(self.ty) + " (mm)\n"
        txt += "tz: " + str(self.tz) + " (mm)\n"
        txt += "rx: " + str(self.rx) + "(degrees)\n"
        txt += "ry: " + str(self.ry) + "(degrees)\n"
        txt += "rz: " + str(self.rz)+ " (degrees)\n"
        txt += "sx: " + str(self.sx) + "\n"
        txt += "sy: " + str(self.sy) + "\n"
        txt += "sz: " + str(self.sz)+ " \n"
        txt += "sxy: " + str(self.sxy) + "\n"
        txt += "syz: " + str(self.syz) + "\n"
        txt += "sxz: " + str(self.sxz)+ " \n"   
        return txt

    def matrix( self ):
        """
        Return matrix.
        """
        return _irtk.affine_matrix( self.tx, self.ty, self.tz,
                                    self.rx, self.ry, self.rz,
                                    self.sx, self.sy, self.sz,
                                    self.sxy, self.syz, self.sxz )
        
    def __from_matrix( self, m ):
        return _irtk.affine_from_matrix(m)

    def write( self, filename ):
        """
        Save to disk.
        """
        _irtk.write_affine( filename,
                            self.tx, self.ty, self.tz,
                            self.rx, self.ry, self.rz,
                            self.sx, self.sy, self.sz,
                            self.sxy, self.syz, self.sxz )

    def apply( self, img, target_header=None,
               interpolation='linear', gaussian_parameter=1.0 ):
        """
        Apply.
        """
        if not isinstance( img, irtk.Image ):
            # point transformation
            pt = np.array(img, dtype='float64', copy=True)
            if len(pt.shape) == 1:
                tmp_pt = np.hstack((pt,[1])).astype('float64')
                return np.dot( self.matrix(), tmp_pt )[:3]
            else:
                pt = _irtk.transform_points( self.matrix(), pt )
                return pt  

        if target_header is None:
            (x_min, y_min, z_min, x_max, y_max, z_max ) = img.bbox(world=True)
            corners = [[x_min, y_min, z_min],
                       [x_max, y_min, z_min],
                       [x_min, y_max, z_min],
                       [x_min, y_min, z_max],
                       [x_max, y_max, z_min],
                       [x_min, y_max, z_max],
                       [x_max, y_min, z_max],
                       [x_max, y_max, z_max]]
            corners = self.apply( corners )
            x_min, y_min, z_min = corners.min(axis=0)
            x_max, y_max, z_max = corners.max(axis=0)
            res = img.header['pixelSize'][0]
            pixelSize = [res, res, res, 1]
            origin = [ x_min + (x_max+1 - x_min)/2,
                       y_min + (y_max+1 - y_min)/2,
                       z_min + (z_max+1 - z_min)/2,
                       img.header['origin'][3] ]
            dim = [ (x_max+1 - x_min)/res,
                    (y_max+1 - y_min)/res,
                    (z_max+1 - z_min)/res,
                    1 ]
            target_header = irtk.new_header( pixelSize=pixelSize, origin=origin, dim=dim)
        if isinstance( target_header, irtk.Image ):
            target_header = target_header.get_header()
        data = img.get_data('float32','cython')
        new_data = _irtk.transform_affine( self.tx, self.ty, self.tz,
                                           self.rx, self.ry, self.rz,
                                           self.sx, self.sy, self.sz,
                                           self.sxy, self.syz, self.sxz,
                                           data,
                                           img.get_header(),
                                           target_header,
                                           interpolation,
                                           gaussian_parameter )
        return irtk.Image( new_data, target_header )

    def invert( self ):
        """
        Invert.
        """
        return AffineTransformation( matrix=np.linalg.inv(self.matrix()) )

    def __mul__(self, other):
        """
        Overloading multiply operator to simulate the composition of transformations:
        returns self * other
        """
        return AffineTransformation( matrix=np.dot(self.matrix(),
                                                   other.matrix()) )

def read_points( filename ):
    """
    Read points from file.
    """
    return np.array( _irtk.read_points(filename),
                     dtype="float64" )

def registration_rigid_points( source, target, rms=False ):
    """
    Point registration.
    """
    source = np.array( source, dtype="float64" )
    target = np.array( target, dtype="float64" )
    ( tx, ty, tz, rx, ry, rz ), RMS = _irtk.registration_rigid_points( source,
                                                                       target )
    t = RigidTransformation( tx=tx, ty=ty, tz=tz,
                             rx=rx, ry=ry, rz=rz )

    if rms:
        return t, RMS
    else:
        return t

def registration_affine_points( source, target, rms=False ):
    """
    Point registration.
    """
    source = np.array( source, dtype="float64" )
    target = np.array( target, dtype="float64" )
    ( tx, ty, tz,
      rx, ry, rz,
      sx, sy, sz,
      sxy, syz, sxz ), RMS = _irtk.registration_affine_points( source,
                                                               target )
    t = AffineTransformation( tx=tx, ty=ty, tz=tz,
                              rx=rx, ry=ry, rz=rz,
                              sx=sx, sy=sy, sz=sz,
                              sxy=sxy, syz=syz, sxz=sxz )

    if rms:
        return t, RMS
    else:
        return t

def flirt2dof( flirt_matrix,
               image_target,
               image_source,
               return_matrix=False ):
    """
    BUG? https://gitlab.doc.ic.ac.uk/sk1712/irtk/commit/343536a64678ba4edaf926b235a5a29a99b90392?view=parallel

    See fsl/src/newimage/newimagefns.cc : raw_affine_transform()
    
    flirt_matrix = np.loadtxt(r)
    https://gitlab.doc.ic.ac.uk/sk1712/irtk/blob/develop/packages/applications/flirt2dof.cc
    """
    
    w2iTgt = image_target.W2I
    i2wSrc = image_source.I2W

    if image_target.order() == "neurological":
        Fy = np.eye( 4, dtype='float64' )
        Fy[0, 0] = -1		
        Fy[0, 3] = image_target.header['dim'][0]-1
        
        w2iTgt = np.dot( Fy, w2iTgt )    
        
    if image_source.order() == "neurological":
        Fy = np.eye( 4, dtype='float64' )
        Fy[0, 0] = -1		
        Fy[0, 3] = image_source.header['dim'][0]-1
        i2wSrc = np.dot( i2wSrc, Fy ) 

    pixelSize_target = image_target.header['pixelSize'].copy()
    pixelSize_source = image_source.header['pixelSize'].copy()

    pixelSize_target[3] = 1
    pixelSize_source[3] = 1

    samplingTarget = np.diag(pixelSize_target)
    samplingSource = np.diag(pixelSize_source)

    flirt_matrix = np.linalg.inv(flirt_matrix)
    samplingSource = np.linalg.inv(samplingSource)
    
    irtk_matrix = np.dot( i2wSrc,
                          np.dot( samplingSource,
                                  np.dot( flirt_matrix,
                                          np.dot( samplingTarget,
                                                  w2iTgt ) ) ) )

    irtk_matrix = np.linalg.inv(irtk_matrix)
    
    if return_matrix:
        return irtk_matrix
    else:
        return AffineTransformation( matrix=irtk_matrix )
   
def flirt_registration( img_src,
                        img_tgt,
                        FSL_BIN="/vol/vipdata/packages/fsl-5.0.1/bin/",
                        dof=7,
                        coarsesearch=60,
                        finesearch=18,
                        verbose=False,
                        cost="corratio"):
    flirt = FSL_BIN+"flirt"
    fh1, source_file = tempfile.mkstemp(suffix=".nii.gz")
    fh2, target_file = tempfile.mkstemp(suffix=".nii.gz")
    fh3, flirt_output = tempfile.mkstemp(suffix=".txt")

    irtk.imwrite( source_file, img_src )
    irtk.imwrite( target_file, img_tgt )
    
    cmd = [ flirt,
            "-cost", cost,
            "-searchcost", cost,
            "-coarsesearch", str(coarsesearch),
            "-finesearch", str(finesearch),
            "-searchrx", "-180", "180",
            "-searchry", "-180", "180",
            "-searchrz", "-180", "180",
            "-dof", str(dof),
            "-in", source_file,
            "-ref", target_file,
            # "-inweight", "tmp/"+patient_id+"_img.nii.gz",
            # "-refweight", "tmp/"+patient_id+"_atlas.nii.gz",
            #"-schedule", "simple3D.sch",
            "-omat", flirt_output ]
    
    utils.run_cmd(cmd, verbose=verbose)

    flirt_matrix = np.loadtxt(flirt_output)
    transformation = flirt2dof( flirt_matrix,
                                img_tgt,
                                img_src,
                                return_matrix=False )
    # clean
    os.close(fh1)
    os.close(fh2)
    os.close(fh3)
    os.remove(flirt_output)
    os.remove(source_file)
    os.remove(target_file)
    
    return transformation
