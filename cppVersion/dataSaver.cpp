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
#include <sstream>
#include "dataSaver.h"
#include "esriHeader.h"

dataSaver::dataSaver(std::string fileName, std::deque <double> > *buffer, boost::condition_variable_any *buffer_available,
		 boost::mutex *buffer_lock)
{
	this -> fileName = fileName;
	this -> buffer = buffer;
	this -> buffer_available = buffer_available;
	this -> buffer_lock = buffer_lock;
	openFile();
}


void dataSaver::openFile()
{
	//open outputfile
	outFile.open(fileName.c_str())
	if (!outFile.is_open())
	{
		std::cerr << "File failed to open" << endl;
		exit(1);	
	}
	//write header info to outputfile
	outFile << "ncols "        << esriHeader.ncols     << endl;
	outFile << "nrows "        << esriHeader.nrows     << endl;
	outFile << "xllcorner "    << esriHeader.xllcorner << endl;
	outFile << "yllcorner "    << esriHeader.yllcorner << endl;
	outFile << "cellsize "     << esriHeader.cellsize  << endl;
	outFile << "NODATA_value " << esriHeader.NODATA    << endl;

	write_func(&outFile);
}

//write data to the output file
void dataSaver::write_func(ofStream* outFile)
{
	deque< deque <double> >* cur_lines = new deque< deque < double> >;
	int count = 0;
	int i;
	//first push back NODATA row for first row
	curr_lines -> push_back(deque<double> (esriHeader.ncols, esriHeader.NODATA));
	//next, grab first two rows of data
	for(i = 0; i < 2; i++)
	{
		//LOCK
		boost::mutex::scoped_lock lock(buffer_lock);
		while(buffer.size() == 0)
		{
			buffer_available.wait(buffer_lock);
		}
		//don't pop anything from curr_lines yet, need to fill with three rows
		curr_lines -> push_back(buffer.front());
		buffer.pop_front();
		buffer_available.notify_one();
		buffer_lock.unlock();
		//UNLOCK
		count++;
	}
	//write out the first row
	for(i = 0; i < esriHeader.ncols; i++)
	{
			*outFile << 
	}
	*outFile << endl;
	//enter main while loop
	while(count < esriHeader.nrows)
	{
		curr_lines -> pop_front();
		//LOCK
		boost::mutex::scoped_lock lock(buffer_lock);
		while(buffer.size() == 0)
		{
			buffer_available.wait(buffer_lock);
		}
		curr_lines -> puch_back(buffer.front());
		buffer.pop_front();
		buffer_available.notify_one();
		buffer_lock.unlock();
		//UNLOCK
		count++;
		for(i = 0; i < esriheaderncols; i++)
		{
			*outFile << 
		}
		*outFile << endl;
	}
	//push back another NODATA row to write the last row
	curr_lines -> pop_front();
	curr_lines -> push_back(deque<double> (esriHeader.ncols, esriHeader.NODATA));
	//write out the last row
	for(i = 0; i < esriHeader.ncols; i++)
	{
		*outFile <<
	}
	*outFile << endl;
}
