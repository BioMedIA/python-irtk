#ifndef VOXELLISE_H
#define VOXELLISE_H

#include "irtk2cython.h"

#include <irtkImage.h>
#include <irtkGaussianBlurring.h>

#ifdef HAS_VTK

#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkStripper.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkImageStencilData.h>
#include <vtkImageStencil.h>
#include <vtkImageData.h>
#include <vtkTriangleFilter.h>
#include <vtkStructuredPoints.h>

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>

#include <vtkRegularPolygonSource.h>
#include <vtkSphereSource.h>
#include <vtkSmoothPolyDataFilter.h>

#include <vtkXMLPolyDataWriter.h>
#include <vtkPolyDataWriter.h>
#include <vtkGenericDataObjectReader.h>

void voxellise( vtkPolyData *poly, // input mesh (must be closed)
                irtkGenericImage<uchar> &image,
                double value=1 );

void create_polydata( double* points,
                      int npoints,
                      int* triangles,
                      int ntriangles,
                      vtkPolyData *poly );

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
                 int* dim );

void _shrinkDisk( uchar* img,
                  int shape0,
                  int shape1,
                  double* center,
                  double radius,
                  int steps );

void _shrinkSphere( double* points,
                    int npoints,
                    uchar* img,
                    double* pixelSize,
                    double* xAxis,
                    double* yAxis,
                    double* zAxis,
                    double* origin,
                    int* dim );

void _read_polydata( char* inputFilename,
                      std::vector< std::vector<double> >& points,
                     std::vector< std::vector<int> >& triangles );
#endif
