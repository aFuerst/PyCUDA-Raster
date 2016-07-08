#ifndef ESRI_HEADER
#define ESRI_HEADER

typedef struct {
    long long ncols;
    long long nrows;
    long long NODATA;
    double cellsize;
    double xllcorner;
    double yllcorner;
} esriHeader;

#endif
