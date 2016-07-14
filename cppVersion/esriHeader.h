#ifndef ESRI_HEADER
#define ESRI_HEADER

#define MAX_BUFF_SIZE 70000

typedef struct {
    long long ncols;
    long long nrows;
    long long NODATA;
    double cellsize;
    double xllcorner;
    double yllcorner;
} esriHeader;

#endif
