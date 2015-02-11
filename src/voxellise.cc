#include "voxellise.h"

#ifdef HAS_VTK

void polydata_WorldToImage( vtkPolyData *poly, 
                             irtkGenericImage<uchar> &image ) {
    double pt[3];

    int noOfPts = poly->GetNumberOfPoints();
    for (int i = 0; i < noOfPts; ++i){
        poly->GetPoints()->GetPoint(i, pt);
        image.WorldToImage( pt[0], pt[1], pt[2] );
        poly->GetPoints()->SetPoint(i, pt);
    }
    
}

void voxellise( vtkPolyData *poly, // input mesh (must be closed)
                irtkGenericImage<uchar> &image,
                double value ) {

    vtkSmartPointer<vtkImageData> whiteImage = vtkSmartPointer<vtkImageData>::New();    
    double spacing[3]; 
    spacing[0] = 1.0;
    spacing[1] = 1.0;
    spacing[2] = 1.0;
    whiteImage->SetSpacing(spacing);
    whiteImage->SetExtent( 0, image.GetX() - 1,
                           0, image.GetY() - 1,
                           0, image.GetZ() - 1 );
    double origin[3];
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    whiteImage->SetOrigin(origin);
    whiteImage->SetScalarTypeToUnsignedChar();
    whiteImage->AllocateScalars();

    for ( int i = 0; i < whiteImage->GetNumberOfPoints(); i++ )
        whiteImage->GetPointData()->GetScalars()->SetTuple1(i, value);  

    // polygonal data --> image stencil:
    vtkSmartPointer<vtkPolyDataToImageStencil> pol2stenc = 
        vtkSmartPointer<vtkPolyDataToImageStencil>::New();
    pol2stenc->SetInput( poly );
    pol2stenc->SetOutputOrigin(origin);
    pol2stenc->SetOutputSpacing(spacing);
    pol2stenc->SetTolerance(0.0);
    pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
    pol2stenc->Update();

    // cut the corresponding white image and set the background:
    vtkSmartPointer<vtkImageStencil> imgstenc = 
        vtkSmartPointer<vtkImageStencil>::New();
    imgstenc->SetInput(whiteImage);
    imgstenc->SetStencil(pol2stenc->GetOutput());
    imgstenc->ReverseStencilOff();
    imgstenc->SetBackgroundValue(0);
    imgstenc->Update();

    vtkSmartPointer<vtkImageData> vtkimageOut =
        vtkSmartPointer<vtkImageData>::New();
    vtkimageOut = imgstenc->GetOutput();
    vtkimageOut->Modified();
    vtkimageOut->Update();

    // Retrieve the output in IRTK
    int n    = image.GetNumberOfVoxels();
    uchar* ptr1 = image.GetPointerToVoxels();
    uchar* ptr2 = (uchar*)vtkimageOut->GetScalarPointer();
    for ( int i = 0; i < n; i++ ){
        *ptr1 = *ptr2;
        ptr1++;
        ptr2++;
    }    
}

void create_polydata( double* points,
                      int npoints,
                      int* triangles,
                      int ntriangles,
                      vtkPolyData *poly ) {

    int i;

    // Setup points
    vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    for ( i = 0; i < npoints; i++ )
        vtk_points->InsertNextPoint( points[index(i, 0, npoints, 3)],
                                 points[index(i, 1, npoints, 3)],
                                 points[index(i, 2, npoints, 3)] );
 
    // Setup triangles
    vtkSmartPointer<vtkCellArray> vtk_triangles = vtkSmartPointer<vtkCellArray>::New();
    for ( i = 0; i < ntriangles; i++ ) {
        vtkSmartPointer<vtkTriangle> vtk_triangle = vtkSmartPointer<vtkTriangle>::New();
        vtk_triangle->GetPointIds()->SetId(0, triangles[index(i, 0, ntriangles, 3)]);
        vtk_triangle->GetPointIds()->SetId(1, triangles[index(i, 1, ntriangles, 3)]);
        vtk_triangle->GetPointIds()->SetId(2, triangles[index(i, 2, ntriangles, 3)]);
        vtk_triangles->InsertNextCell(vtk_triangle);
    }

    poly->SetPoints(vtk_points);
    poly->SetPolys(vtk_triangles);
    poly->Update();
}



#endif

void _voxellise( double* points,
                 int npoints,
                 int* triangles,
                 int ntriangles,
                 uchar* img,
                 double* pixelSize,
                 double* xAxis,
                 double* yAxis,
                 double* zAxis,
                 double* origin,
                 int* dim ) {
#ifdef HAS_VTK
    vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();

    create_polydata( points,
                     npoints,
                     triangles,
                     ntriangles,
                     poly );

    

    // Write the file
    vtkSmartPointer<vtkPolyDataWriter> writer =  
        vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName("debug.vtk");
    writer->SetInput(poly);
    writer->Write();

    irtkGenericImage<uchar> irtk_image;
    py2irtk<uchar>( irtk_image,
                    img,
                    pixelSize,
                    xAxis,
                    yAxis,
                    zAxis,
                    origin,
                    dim );

    polydata_WorldToImage( poly, irtk_image );

    voxellise( poly,
               irtk_image,
               1 );

    irtk2py<uchar>( irtk_image,
                    img,
                    pixelSize,
                    xAxis,
                    yAxis,
                    zAxis,
                    origin,
                    dim );
#endif
}

void _shrinkDisk( uchar* img,
                  int shape0,
                  int shape1,
                  double* center,
                  double radius,
                  int steps ) {
    
#ifdef HAS_VTK
        
    //double pt[3];
        
    vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> vtk_verts = vtkSmartPointer<vtkCellArray>::New();
    for ( int y = 0; y < shape0; y++ )
        for ( int x = 0; x < shape1; x++ )
            if ( img[index(y,x,shape0,shape1)] > 0 ) {
                // pt[0] = x;
                // pt[1] = y;
                // pt[3] = 0;
                //irtk_image.ImageToWorld( pt[0], pt[1], pt[2] );
                vtk_verts->InsertNextCell(1);
                vtk_verts->InsertCellPoint( vtk_points->InsertNextPoint( x,
                                                                         y,
                                                                         0 ) );
        }
    //pt[0] = 0;
    //irtk_image.ImageToWorld( center[0], center[1], pt[0] );
    vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(vtk_points);
    poly->SetVerts(vtk_verts);
    poly->Update();

    // Create a circle
    vtkSmartPointer<vtkRegularPolygonSource> circle =
        vtkSmartPointer<vtkRegularPolygonSource>::New();
 
    circle->GeneratePolygonOff();
    circle->GeneratePolylineOn();
    circle->SetNumberOfSides( steps );
    circle->SetRadius( radius );
    circle->SetCenter( center[1], center[0], 0 );
    circle->Update();
    
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter =
        vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoothFilter->SetInputConnection( circle->GetOutputPort() );
    smoothFilter->SetSource( poly );
    //smoothFilter->SetNumberOfIterations(10);
    smoothFilter->SetEdgeAngle( 180.0 );
    smoothFilter->FeatureEdgeSmoothingOn();
        smoothFilter->SetFeatureAngle(180.0);
    smoothFilter->Update();
    std::cout << "edge angle " << smoothFilter->GetEdgeAngle()<<"\n";

    //     vtkSmartPointer<vtkPolyDataWriter> writer =  
    //     vtkSmartPointer<vtkPolyDataWriter>::New();
    // writer->SetFileName("debug.vtk");
    // writer->SetInput(circle->GetOutput());
    // writer->Write();

irtkGenericImage<uchar> irtk_image( shape1, shape0, 1 );
    //irtk_image = 0;

    

 voxellise( //circle->GetOutput(),
               smoothFilter->GetOutput(),
               irtk_image,
               1 );

    irtk_image.Write( "irtk_image.nii");

    irtk2py_buffer<uchar>( irtk_image, img );
#endif
}

void _shrinkSphere( double* points,
                    int npoints,
                    uchar* img,
                    double* pixelSize,
                    double* xAxis,
                    double* yAxis,
                    double* zAxis,
                    double* origin,
                    int* dim ) {
    
#ifdef HAS_VTK

    int i;
    double x, y, z;
    double cx, cy, cz;
    double r;
    double d;

    cx = 0; cy = 0; cz = 0; r = 0;
    
    // Setup points
    int id;
    vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> vtk_vertices = vtkSmartPointer<vtkCellArray>::New();
    for ( i = 0; i < npoints; i++ ) {
        x = points[index(i, 0, npoints, 3)];
        y = points[index(i, 1, npoints, 3)];
        z = points[index(i, 2, npoints, 3)];
        id = vtk_points->InsertNextPoint( x, y, z );
        vtk_vertices->InsertNextCell(1);
        vtk_vertices->InsertCellPoint(id);
        cx += x;
        cy += y;
        cz += z;
    }
    cx /= npoints;
    cy /= npoints;
    cz /= npoints;

    for ( i = 0; i < npoints; i++ ) {
        x = points[index(i, 0, npoints, 3)];
        y = points[index(i, 1, npoints, 3)];
        z = points[index(i, 2, npoints, 3)];
        d = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
        if (d>r)
            r = d;
    }

    vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(vtk_points);
    poly->SetPolys(vtk_vertices);
    poly->Update();
    
    vtkSmartPointer<vtkSphereSource> sphere =
        vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetCenter(cx,cy,cz);
    sphere->SetRadius(r);
    sphere->SetPhiResolution(100);
    sphere->SetThetaResolution(100);
    sphere->Update();
    
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = 
        vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoothFilter->SetInputConnection(0, sphere->GetOutputPort());
    smoothFilter->SetInput(1, poly);
    smoothFilter->Update();

    poly = smoothFilter->GetOutput();

    irtkGenericImage<uchar> irtk_image;
    py2irtk<uchar>( irtk_image,
                    img,
                    pixelSize,
                    xAxis,
                    yAxis,
                    zAxis,
                    origin,
                    dim );

    polydata_WorldToImage( poly, irtk_image );
    
    voxellise( poly,
               irtk_image,
               1 );

    irtk2py<uchar>( irtk_image,
                    img,
                    pixelSize,
                    xAxis,
                    yAxis,
                    zAxis,
                    origin,
                    dim );   
#endif
}

void _read_polydata( char* inputFilename,
                     std::vector< std::vector<double> >& points,
                     std::vector< std::vector<int> >& triangles ) {
#ifdef HAS_VTK
    // Get all data from the file
    vtkSmartPointer<vtkGenericDataObjectReader> reader = 
        vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFilename);
    reader->Update();
 
    // All of the standard data types can be checked and obtained like this:
    if(! reader->IsFilePolyData() ) {
        std::cout << "output is not a polydata" << std::endl;
    }
    vtkSmartPointer<vtkPolyData> poly = reader->GetPolyDataOutput();

    // http://www.vtk.org/pipermail/vtkusers/2007-September/042897.html
    for(vtkIdType i = 0; i < poly->GetNumberOfPoints(); i++) {
        double p[3];
        poly->GetPoint(i,p);
        std::vector<double> point(3);
        point[0] = p[0];
        point[1] = p[1];
        point[2] = p[2];
        points.push_back(point);
    }

    vtkIdType npts, *pts;
    poly->GetPolys()->InitTraversal();
    while ( poly->GetPolys()->GetNextCell(npts,pts) ) {
        std::vector<int> triangle(3);
        triangle[0] = pts[0];
        triangle[1] = pts[1];
        triangle[2] = pts[2];
        triangles.push_back(triangle);
    }

    return;

#endif
}
