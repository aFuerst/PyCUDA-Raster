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
        /*
            Constructor for dataLoader  
            parameters: 
                _fileName - path to asc file to be used as input
                _buffer - pointer to shared buffer to pass data into    
                _buffer_available - boost condition to notify when buffer is available
                _buffer_lock - lock on shared buffer to ensure thread-safe access
        */
        dataLoader(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available, boost::mutex *buffer_lock);

        /*
            Simple destructor for dataLoader
            just closes input file
        */
        ~dataLoader();

        /*
            return header info as a string
        */
        std::string getHeader(void);


        /*
            Opens file, reads header, and starts reading input file
        */
        void run();

    private:

        /*
            Reads the first 6 lines of the header to get raster information
            store them in instance vars. Requires file to already be opened
            Advances the file pointer to the first line of actual data
        */
        void readHeader();

        /*
            Reads inFile one line at a time and passes the double version of 
            the asc line into its shaed buffer. Assumes the file pointer    
            is at the first line of data 
        */
        void readFile();

        /*
            Opens fileName given to constructor and puts ifstream object in inFile
        */
        void openFile();

        /* header information */
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
