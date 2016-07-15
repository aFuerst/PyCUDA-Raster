#ifndef DATASAVER
#define DATASAVER

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
#include "esriHeader.h"

class dataSaver
{
public:

    /*
        Constructor for dataSaver
        parameters: 
            _fileName - path to asc file to be used as output
            _buffer - pointer to shared buffer to pass data into    
            _buffer_available - boost condition to notify when buffer is available
            _buffer_lock - lock on shared buffer to ensure thread-safe access
            _header - esriHeader object with header infotmation to write at beginning of file
    */
	dataSaver(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available,
		 boost::mutex *buffer_lock, esriHeader* header);

    /*
        Deconstructor, closes output file
    */
    ~dataSaver();

    /*
        Opens file and begins write_func to print data to file
    */
    void run();

private:

    /*
        Opens file and writes header information to it
    */
	void openFile();

    /*
        Write data to the output file a single line at a time from its input buffer 
    */
	void write_func();

	std::string fileName;
	std::ofstream outFile;	
	std::deque<std::deque <double> > *buffer;
	boost::condition_variable_any *buffer_available;
	boost::mutex *buffer_lock;
    esriHeader* header;
};

#endif
