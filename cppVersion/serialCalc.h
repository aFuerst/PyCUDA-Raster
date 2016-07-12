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
    serialCalc(std::deque< std::deque <double> >* loadBuffer, std::vector< std::string >* functions, esriHeader* header,
               boost::condition_variable_any* load_buffer_available, boost::mutex* load_buffer_lock, 
               std::vector< std::deque< std::deque <double> >* >* outBuffers, std::vector< boost::condition_variable_any* >* buffer_available_list,
               std::vector< boost::mutex* >* buffer_lock_list);
    void run();

private:
    std::vector< std::string >* functions;

    std::deque<std::deque <double> > *input_buffer;
    boost::condition_variable_any *input_buffer_available;
    boost::mutex *input_buffer_lock;

    std::vector< std::deque< std::deque <double> >* >* outBuffers;
    std::vector< boost::condition_variable_any* >* buffer_available_list;
    std::vector< boost::mutex* >* buffer_lock_list;

    esriHeader header_info;

    double calculate(std::deque< std::deque <double> >* cur_lines, int i, std::string function);
    double calc_slope(std::deque< std::deque <double> >* cur_lines, int col);
    double calc_aspect();
    double calc_hillshade();
    void run_func();
};

#endif
