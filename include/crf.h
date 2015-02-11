#include "irtk2cython.h"
#include "irtkCRF.h"
#include "irtkSimpleGraphcut.h"

#define PY_FORMAT_LONG_LONG "ll"

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
           double sigmaZ );

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
                double sigmaZ );
