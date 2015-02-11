#include "registration.h"

void rigid2py( irtkRigidTransformation &transform,
               double &tx,
               double &ty,
               double &tz,
               double &rx,
               double &ry,
               double &rz,
               bool invert ) {
    if (invert) {
        transform.Invert(); // inverts only the matrix
        transform.UpdateParameter();
    }
    tx = transform.GetTranslationX();
    ty = transform.GetTranslationY();
    tz = transform.GetTranslationZ();
    rx = transform.GetRotationX();
    ry = transform.GetRotationY();
    rz = transform.GetRotationZ();
}

void py2rigid( irtkRigidTransformation &transform,
               double tx,
               double ty,
               double tz,
               double rx,
               double ry,
               double rz,
               bool invert ) {
    transform.PutTranslationX( tx );
    transform.PutTranslationY( ty );
    transform.PutTranslationZ( tz );
    transform.PutRotationX( rx );
    transform.PutRotationY( ry );
    transform.PutRotationZ( rz );
    if (invert) {
        transform.Invert(); // inverts only the matrix
        transform.UpdateParameter();
    }    
}

void affine2py( irtkAffineTransformation &transform,
                double &tx,
                double &ty,
                double &tz,
                double &rx,
                double &ry,
                double &rz,
                double &sx,
                double &sy,
                double &sz,
                double &sxy,
                double &syz,
                double &sxz,
                bool invert ) {
    if (invert) {
        transform.Invert(); // inverts only the matrix
        transform.UpdateParameter();
    }
    tx = transform.GetTranslationX();
    ty = transform.GetTranslationY();
    tz = transform.GetTranslationZ();
    rx = transform.GetRotationX();
    ry = transform.GetRotationY();
    rz = transform.GetRotationZ();
    sx = transform.GetScaleX();
    sy = transform.GetScaleY();
    sz = transform.GetScaleZ();
    sxy = transform.GetShearXY();
    syz = transform.GetShearYZ();
    sxz = transform.GetShearXZ();
}

void py2affine( irtkAffineTransformation &transform,
                double tx,
                double ty,
                double tz,
                double rx,
                double ry,
                double rz,
                double sx,
                double sy,
                double sz,
                double sxy,
                double syz,
                double sxz,
                bool invert ) {
    transform.PutTranslationX( tx );
    transform.PutTranslationY( ty );
    transform.PutTranslationZ( tz );
    transform.PutRotationX( rx );
    transform.PutRotationY( ry );
    transform.PutRotationZ( rz );
    transform.PutScaleX( sx );
    transform.PutScaleY( sy );
    transform.PutScaleZ( sz );
    transform.PutShearXY( sxy );
    transform.PutShearYZ( syz );
    transform.PutShearXZ( sxz );
    if (invert) {
        transform.Invert(); // inverts only the matrix
        transform.UpdateParameter();
    }    
}

void pyList2rigidVector( std::vector< irtkRigidTransformation > &vec,
                         double* tx,
                         double* ty,
                         double* tz,
                         double* rx,
                         double* ry,
                         double* rz,
                         int n,
                         bool invert ) {

    // clean the vector first
    vec.clear();
    vec.reserve( n );

    for ( int i = 0; i < n; i++ ) {
        irtkRigidTransformation transform;
        py2rigid( transform,
                  tx[i],
                  ty[i],
                  tz[i],
                  rx[i],
                  ry[i],
                  rz[i],
                  invert );
        vec.push_back(transform);
    }    
}

void rigidVector2pyList( std::vector< irtkRigidTransformation > &vec,
                         double* tx,
                         double* ty,
                         double* tz,
                         double* rx,
                         double* ry,
                         double* rz,
                         bool invert ) {

    int n = vec.size();

    for ( int i = 0; i < n; i++ )
        rigid2py( vec[i],
                  tx[i],
                  ty[i],
                  tz[i],
                  rx[i],
                  ry[i],
                  rz[i],
                  invert );
 
}

void _read_rigid( char* filename,
                  double &tx,
                  double &ty,
                  double &tz,
                  double &rx,
                  double &ry,
                  double &rz ) {
    irtkRigidTransformation transform;
    transform.irtkTransformation::Read( filename );
    rigid2py( transform,
              tx,ty,tz,
              rx ,ry, rz );
}

void _write_rigid( char* filename,
                   double tx,
                   double ty,
                   double tz,
                   double rx,
                   double ry,
                   double rz ) {
    irtkRigidTransformation transform;
    py2rigid( transform,
              tx,ty,tz,
              rx ,ry, rz );
    transform.irtkTransformation::Write( filename );
}

void _transform_rigid( double tx,
                       double ty,
                       double tz,
                       double rx,
                       double ry,
                       double rz,
                       float* source_img,
                       double* source_pixelSize,
                       double* source_xAxis,
                       double* source_yAxis,
                       double* source_zAxis,
                       double* source_origin,
                       int* source_dim,
                       float* target_img,
                       double* target_pixelSize,
                       double* target_xAxis,
                       double* target_yAxis,
                       double* target_zAxis,
                       double* target_origin,
                       int* target_dim,
                       int interpolation_method,
                       float gaussian_parameter ) {

    // transformation
    irtkRigidTransformation transform;
    py2rigid( transform,
              tx,ty,tz,
              rx ,ry, rz,
              true );

    // source
    irtkGenericImage<float> source;
    py2irtk<float>( source,
                    source_img,
                    source_pixelSize,
                    source_xAxis,
                    source_yAxis,
                    source_zAxis,
                    source_origin,
                    source_dim );

    // target
    irtkGenericImage<float> target;   
    irtkImageAttributes attr;
    put_attributes( attr,
                    target_pixelSize,
                    target_xAxis,
                    target_yAxis,
                    target_zAxis,
                    target_origin,
                    target_dim );
    target.Initialize( attr );

    // interpolator
    irtkImageFunction *interpolator = NULL;

    switch (interpolation_method) {
          
    case NEAREST_NEIGHBOR:
        { interpolator = new irtkNearestNeighborInterpolateImageFunction; }
        break;

    case LINEAR:
        { interpolator = new irtkLinearInterpolateImageFunction; }
        break;

    case BSPLINE:
        { interpolator = new irtkBSplineInterpolateImageFunction; }
        break;

    case CSPLINE:
        { interpolator = new irtkCSplineInterpolateImageFunction; }
        break;

    case SINC:
        { interpolator = new irtkSincInterpolateImageFunction; }
        break;

    case SHAPE:
        { interpolator = new irtkShapeBasedInterpolateImageFunction; }
        break;

    case GAUSSIAN:
        { interpolator = new irtkGaussianInterpolateImageFunction(gaussian_parameter); }
        break;

    default:
        cout << "Unknown interpolation method" << endl;
    }

    // Create image transformation
    irtkImageTransformation imagetransformation;

    imagetransformation.SetInput( &source, &transform );
    imagetransformation.SetOutput( &target );
    imagetransformation.PutInterpolator( interpolator );

    // padding
    int target_padding, source_padding;
    source_padding = 0;
    target_padding = MIN_GREY;
    imagetransformation.PutTargetPaddingValue( target_padding );
    imagetransformation.PutSourcePaddingValue( source_padding );

    // inverse transformation
    // imagetransformation.InvertOn();
    
    // Transform image
    imagetransformation.Run();

    irtk2py<float>( target,
                    target_img,
                    target_pixelSize,
                    target_xAxis,
                    target_yAxis,
                    target_zAxis,
                    target_origin,
                    target_dim );

    delete interpolator;
}

void _registration_rigid( short* source_img,
                          double* source_pixelSize,
                          double* source_xAxis,
                          double* source_yAxis,
                          double* source_zAxis,
                          double* source_origin,
                          int* source_dim,
                          short* target_img,
                          double* target_pixelSize,
                          double* target_xAxis,
                          double* target_yAxis,
                          double* target_zAxis,
                          double* target_origin,
                          int* target_dim,
                          double &tx,
                          double &ty,
                          double &tz,
                          double &rx,
                          double &ry,
                          double &rz ) {
    
    // source
    /** Second input image. This image is denoted as source image. The goal of
     *  the registration is to find the transformation which maps the source
     *  image into the coordinate system of the target image.
     */
    irtkGenericImage<short> source;
    py2irtk<short>( source,
                    source_img,
                    source_pixelSize,
                    source_xAxis,
                    source_yAxis,
                    source_zAxis,
                    source_origin,
                    source_dim );
    

    // target
    /** First set of input image. This image is denoted as target image and its
     *  coordinate system defines the frame of reference for the registration.
     */
    irtkGenericImage<short> target;
    py2irtk<short>( target,
                    target_img,
                    target_pixelSize,
                    target_xAxis,
                    target_yAxis,
                    target_zAxis,
                    target_origin,
                    target_dim );     
    
    // Create transformation
    irtkRigidTransformation transformation;

    // Initialize transformation
    py2rigid( transformation,
              tx,ty,tz,
              rx ,ry, rz );

    // transformation.Invert(); // inverts only the matrix
    // transformation.UpdateParameter();
    
    // Create registration
    // The goal of the registration is to find the transformation which maps the
    // source image into the coordinate system of the target image whereas
    // irtkImageTransformation expect the inverse transformations, hence the
    // calls to Invert().
    irtkImageRigidRegistrationWithPadding registration;
    
    registration.SetInput( &source, &target );
    registration.SetOutput( &transformation );

    // Make an initial Guess for the parameters.
    //registration.GuessParameter();
    registration.GuessParameterSliceToVolume();
    
    // TODO: Overrride with any the user has set.
    // use parameter file?
    
    // Run registration filter
    registration.Run();

    // transformation.Invert(); // inverts only the matrix
    // transformation.UpdateParameter();

    // We return the transformation mapping locations in the target image to
    // locations in the source image (this is the input expected by
    // irtkImageTransformation). 

    rigid2py( transformation,
              tx,ty,tz,
              rx ,ry, rz );
    
}

void _read_affine( char* filename,
                   double &tx,
                   double &ty,
                   double &tz,
                   double &rx,
                   double &ry,
                   double &rz,
                   double &sx,
                   double &sy,
                   double &sz,
                   double &sxy,
                   double &syz,
                   double &sxz ) {
    
    irtkAffineTransformation transform;
    transform.irtkTransformation::Read( filename );
    affine2py( transform,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz );
}

void _transform_affine( double tx,
                        double ty,
                        double tz,
                        double rx,
                        double ry,
                        double rz,
                        double sx,
                        double sy,
                        double sz,
                        double sxy,
                        double syz,
                        double sxz,
                        float* source_img,
                        double* source_pixelSize,
                        double* source_xAxis,
                        double* source_yAxis,
                        double* source_zAxis,
                        double* source_origin,
                        int* source_dim,
                        float* target_img,
                        double* target_pixelSize,
                        double* target_xAxis,
                        double* target_yAxis,
                        double* target_zAxis,
                        double* target_origin,
                        int* target_dim,
                        int interpolation_method,
                        float gaussian_parameter ) {

    // transformation
    irtkAffineTransformation transform;
    py2affine( transform,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz,
               true );

    // source
    irtkGenericImage<float> source;
    py2irtk<float>( source,
                    source_img,
                    source_pixelSize,
                    source_xAxis,
                    source_yAxis,
                    source_zAxis,
                    source_origin,
                    source_dim );

    // target
    irtkGenericImage<float> target;   
    irtkImageAttributes attr;
    put_attributes( attr,
                    target_pixelSize,
                    target_xAxis,
                    target_yAxis,
                    target_zAxis,
                    target_origin,
                    target_dim );
    target.Initialize( attr );

    // interpolator
    irtkImageFunction *interpolator = NULL;

    switch (interpolation_method) {
          
    case NEAREST_NEIGHBOR:
        { interpolator = new irtkNearestNeighborInterpolateImageFunction; }
        break;

    case LINEAR:
        { interpolator = new irtkLinearInterpolateImageFunction; }
        break;

    case BSPLINE:
        { interpolator = new irtkBSplineInterpolateImageFunction; }
        break;

    case CSPLINE:
        { interpolator = new irtkCSplineInterpolateImageFunction; }
        break;

    case SINC:
        { interpolator = new irtkSincInterpolateImageFunction; }
        break;

    case SHAPE:
        { interpolator = new irtkShapeBasedInterpolateImageFunction; }
        break;

    case GAUSSIAN:
        { interpolator = new irtkGaussianInterpolateImageFunction(gaussian_parameter); }
        break;

    default:
        cout << "Unknown interpolation method" << endl;
    }

    // Create image transformation
    irtkImageTransformation imagetransformation;

    imagetransformation.SetInput( &source, &transform );
    imagetransformation.SetOutput( &target );
    imagetransformation.PutInterpolator( interpolator );

    // padding
    int target_padding, source_padding;
    source_padding = 0;
    target_padding = MIN_GREY;
    imagetransformation.PutTargetPaddingValue( target_padding );
    imagetransformation.PutSourcePaddingValue( source_padding );

    // inverse transformation
    // imagetransformation.InvertOn();
    
    // Transform image
    imagetransformation.Run();

    irtk2py<float>( target,
                    target_img,
                    target_pixelSize,
                    target_xAxis,
                    target_yAxis,
                    target_zAxis,
                    target_origin,
                    target_dim );

    delete interpolator;
}

void py2pointSet( irtkPointSet &irtk_points,
                  double* data,
                  int n ) {
    for (int i = 0; i < n; i++) {
        irtk_points.Add( irtkPoint( data[3*i],
                                    data[3*i+1],
                                    data[3*i+2] ) );
  }
}

void pointSet2py( irtkPointSet &irtk_points,
                  double* data,
                  int n ) {
    for (int i = 0; i < n; i++) {
        irtkPoint point = irtk_points(i);
        data[3*i] = point._x;
        data[3*i+1] = point._y;
        data[3*i+2] = point._z;
  }
}

double _registration_rigid_points( double* source_points,
                                   double* target_points,
                                   int n,
                                   double &tx,
                                   double &ty,
                                   double &tz,
                                   double &rx,
                                   double &ry,
                                   double &rz ) {
    double error;
    irtkPointSet target, source;

    py2pointSet( source, source_points, n );
    py2pointSet( target, target_points, n );
    
    // Create registration filter
    irtkPointRigidRegistration registration;

    // Create transformation
    irtkRigidTransformation transformation;
  
    registration.SetInput( &source, &target );
    registration.SetOutput( &transformation );

    // Run registration filter
    registration.Run();

    rigid2py( transformation,
              tx,ty,tz,
              rx ,ry, rz );
    
    // Calculate residual error
    transformation.irtkTransformation::Transform( source );

    error = 0;
    for ( int i = 0; i < target.Size(); i++ ) {
        irtkPoint p1 = target(i);
        irtkPoint p2 = source(i);
        error += sqrt(pow(double(p1._x - p2._x), 2.0) +
                      pow(double(p1._y - p2._y), 2.0) +
                      pow(double(p1._z - p2._z), 2.0));
    }
    return error/target.Size(); // RMS in mm
}

double _registration_affine_points( double* source_points,
                                    double* target_points,
                                    int n,
                                    double &tx,
                                    double &ty,
                                    double &tz,
                                    double &rx,
                                    double &ry,
                                    double &rz,
                                    double &sx,
                                    double &sy,
                                    double &sz,
                                    double &sxy,
                                    double &syz,
                                    double &sxz ) {
    double error;
    irtkPointSet target, source;

    py2pointSet( source, source_points, n );
    py2pointSet( target, target_points, n );
    
    // Create registration filter
    irtkPointAffineRegistration registration;

    // Create transformation
    irtkAffineTransformation transformation;

    if (sxy==-1)
        transformation.PutStatus(SXY, _Passive);
    if (syz==-1)
        transformation.PutStatus(SYZ, _Passive);
    if (sxz==-1)
        transformation.PutStatus(SXZ, _Passive);
    
    registration.SetInput( &source, &target );
    registration.SetOutput( &transformation );

    // Run registration filter
    registration.Run();

    affine2py( transformation,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz );
    
    // Calculate residual error
    transformation.irtkTransformation::Transform( source );

    error = 0;
    for ( int i = 0; i < target.Size(); i++ ) {
        irtkPoint p1 = target(i);
        irtkPoint p2 = source(i);
        error += sqrt(pow(double(p1._x - p2._x), 2.0) +
                      pow(double(p1._y - p2._y), 2.0) +
                      pow(double(p1._z - p2._z), 2.0));
    }
    return error/target.Size(); // RMS in mm
}

void _affine_matrix( double tx,
                     double ty,
                     double tz,
                     double rx,
                     double ry,
                     double rz,
                     double sx,
                     double sy,
                     double sz,
                     double sxy,
                     double syz,
                     double sxz,
                     double* m ) {
    irtkAffineTransformation transform;
    py2affine( transform,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz,
               false );
    irtkMatrix transform_matrix = transform.GetMatrix();
    for ( int i = 0; i < 4; i++ )
        for ( int j = 0; j < 4; j++ ) {
            m[index(i,j,4,4)] = transform_matrix(i,j);
        } 
}

void _affine_from_matrix( double &tx,
                          double &ty,
                          double &tz,
                          double &rx,
                          double &ry,
                          double &rz,
                          double &sx,
                          double &sy,
                          double &sz,
                          double &sxy,
                          double &syz,
                          double &sxz,
                          double* m ) {
    irtkMatrix transform_matrix(4,4);
    for ( int i = 0; i < 4; i++ )
        for ( int j = 0; j < 4; j++ ) {
            transform_matrix(i,j) = m[index(i,j,4,4)];
        }
    irtkAffineTransformation transform;
    transform.PutMatrix(transform_matrix);
    affine2py( transform,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz,
               false );
}

void _write_affine( char* filename,
                    double tx,
                    double ty,
                    double tz,
                    double rx,
                    double ry,
                    double rz,
                    double sx,
                    double sy,
                    double sz,
                    double sxy,
                    double syz,
                    double sxz ) {
    irtkAffineTransformation transform;
    py2affine( transform,
               tx,ty,tz,
               rx ,ry, rz,
               sx, sy, sz,
               sxy, syz, sxz );
    transform.irtkTransformation::Write( filename );
}
