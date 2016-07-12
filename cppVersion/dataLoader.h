#ifndef DATALOADER
#define DATALOADER

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

class dataLoader{

    public:
        dataLoader(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available, boost::mutex *buffer_lock);
        ~dataLoader();
        std::string getHeader(void);
        void run();

    private:
        void readHeader();
        void readFile();
        void openFile();

        long long ncols;
        long long nrows;
        double cellsize;
        double nodata;
        long long xllcorner;
        long long yllcorner;

        std::string fileName;
        std::ifstream inFile;
        std::deque<std::deque <double> > *buffer;
        boost::condition_variable_any *buffer_available;
        boost::mutex *buffer_lock;
};

#endif
