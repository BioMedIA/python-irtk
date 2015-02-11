#include "crf.h" 

void _crf( pixel_t* img,
           double* pixelSize,
           double* xAxis,
           double* yAxis,
           double* zAxis,
           double* origin,
           int* dim,
           LabelID* labels,
           int nb_labels,
           double* proba,
           double l,
           double sigma,
           double sigmaZ ) {

    irtkGenericImage<pixel_t> irtk_image;
    py2irtk<pixel_t>( irtk_image,
                      img,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );

    irtkGenericImage<LabelID> irtk_labels;
    py2irtk<LabelID>( irtk_labels,
                      labels,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );

    dim[3] = nb_labels;
    irtkGenericImage<double> irtk_proba;
    py2irtk<double>( irtk_proba,
                     proba,
                     pixelSize,
                     xAxis,
                     yAxis,
                     zAxis,
                     origin,
                     dim );

    irtkCRF crf( irtk_image,
                 irtk_labels,
                 irtk_proba );
    crf.SetLambda( l );
    crf.SetSigma( sigma );
    crf.SetSigmaZ( sigmaZ );
    crf.Run();

    dim[3] = 1;
    irtk2py<LabelID>( irtk_labels,
                      labels,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );
}

void _graphcut( pixel_t* img,
                double* pixelSize,
                double* xAxis,
                double* yAxis,
                double* zAxis,
                double* origin,
                int* dim,
                LabelID* labels,
                double l,
                double sigma,
                double sigmaZ ) {

    irtkGenericImage<pixel_t> irtk_image;
    py2irtk<pixel_t>( irtk_image,
                      img,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );

    irtkGenericImage<LabelID> irtk_labels;
    py2irtk<LabelID>( irtk_labels,
                      labels,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );

    irtkSimpleGraphcut graphcut( irtk_image,
                                 irtk_labels );
    graphcut.SetSigma( sigma );
    graphcut.SetSigmaZ( sigmaZ );
    graphcut.Run();

    irtk2py<LabelID>( irtk_labels,
                      labels,
                      pixelSize,
                      xAxis,
                      yAxis,
                      zAxis,
                      origin,
                      dim );
}
