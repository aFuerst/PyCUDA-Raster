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
	dataSaver(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available,
		 boost::mutex *buffer_lock, esriHeader* header);
    void run();

private:
	void openFile();
	void write_func();

	std::string fileName;
	std::ofstream outFile;	
	std::deque<std::deque <double> > *buffer;
	boost::condition_variable_any *buffer_available;
	boost::mutex *buffer_lock;
    esriHeader* header;
};

#endif
