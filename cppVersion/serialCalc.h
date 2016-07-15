#ifndef SERIALCALC
#define SERIALCALC

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <boost/thread.hpp>
#include <deque>
#include <sstream>
#include "esriHeader.h"

class serialCalc{

public:

    /*
        Constructor for dataSaver
        parameters: 
            _fileName - path to asc file to be used as output
            _loadBuffer - pointer to shared buffer to receive data from    
            _load_buffer_available - boost condition to notify when load buffer is available
            _load_buffer_lock - lock on shared buffer to ensure thread-safe access
            _header - esriHeader object with header infotmation to write at beginning of file
            _functions - vector of function names to calculate
            _outBuffers - vector of out buffers to pass calculated data to
            _buffer_available_list - vector of boost conditions for outBuffers
            _buffer_lock_list - vector of boost locks for outBuffers
    */
    serialCalc(std::deque< std::deque <double> >* loadBuffer, std::vector< std::string >* functions, esriHeader* header,
               boost::condition_variable_any* load_buffer_available, boost::mutex* load_buffer_lock, 
               std::vector< std::deque< std::deque <double> >* >* outBuffers, std::vector< boost::condition_variable_any* >* buffer_available_list,
               std::vector< boost::mutex* >* buffer_lock_list);

    /*
        Deconstructor, currently does nothing special
    */
    ~serialCalc();

    /*
        Starts everything object needs to do
    */
    void run();

private:
    std::vector< std::string >* functions;

    std::deque<std::deque <double> > *input_buffer;
    boost::condition_variable_any *input_buffer_available;
    boost::mutex *input_buffer_lock;

    std::vector< std::deque< std::deque <double> >* >* outBuffers;
    std::vector< boost::condition_variable_any* >* buffer_available_list;
    std::vector< boost::mutex* >* buffer_lock_list;

    esriHeader* header_info;

    /*
        Determines which GIS function to compute and calls that function
    */
    double calculate(std::deque< std::deque <double> >* cur_lines, int i, std::string function);

    /* calc_slope - calculates the slope for a single cell in a raster
     *              file. cur_lines includes the input row of the cell, and
     *              the rows above and below that cell. col is the column
     *              of the cell we're calculating the slope for.
     */
    double calc_slope(std::deque< std::deque <double> >* cur_lines, int col);

    /*
        calculates aspect for a single cell at col using a 3x3 kernel
    */
    double calc_aspect(std::deque< std::deque <double> >* cur_lines, int col);

    /*
        calculates hillshade for a single cell at col using a 3x3 kernel
    */
    double calc_hillshade(std::deque< std::deque <double> >* cur_lines, int col);  

    /*
        calculates aspect needed for hillshade
    */ 
    double hillshade_aspect(double dz_dx, double dz_dy);

    /*
        saves 3 deques at a time to calculat slope of middle row
        passes that calculated row into output file
    */
    void run_func();
};

#endif
