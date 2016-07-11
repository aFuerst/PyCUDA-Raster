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
    serialCalc(std::deque<std::deque <double> > *input_buffer, boost::condition_variable_any *input_buffer_available, boost::mutex *input_buffer_lock, 
               std::deque<std::deque <double> > *output_buffer, boost::condition_variable_any *output_buffer_available, boost::mutex *output_buffer_lock, esriHeader head_info);
    void run();

private:
    std::deque<std::deque <double> > *input_buffer;
    boost::condition_variable_any *input_buffer_available;
    boost::mutex *input_buffer_lock;
    std::deque<std::deque <double> > *output_buffer;
    boost::condition_variable_any *output_buffer_available;
    boost::mutex *output_buffer_lock;
    esriHeader header_info;

    double calc_slope(std::deque< std::deque <double> >* cur_lines, int col);
    void run_func();
};

#endif
