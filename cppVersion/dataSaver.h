#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <boost/thread.cpp>
#include <deque>

class dataSaver
{
public:
	dataSaver(std::string fileName, std::deque <double> > *buffer, boost::condition_variable_any *buffer_available,
		std::string getHeader(void));

private:
	void openFile();
	void write_func();

	std::string fileName;
	std::ofstream outFile;	
	std::deque<std::deque <double> > *buffer;
	boost::condition_variable_any *buffer_available;
	boost::mutex *buffer_lock;
};
