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
#include <esriheader.h>

class dataSaver
{
public:
	dataSaver(std::string fileName);

private:
	void openFile();
	void write_func();

	std::string fileName;
	std::ofstream outFile;	
};


dataSaver::dataSaver(std::string fileName)
{
	this -> fileName = fileName;
	openFile();
}

void dataSaver::openFile()
{
	outFile.open(fileName.c_str())
	if (!outFile.is_open())
	{
		std::cerr << "File failed to open" << endl;
		exit(1);	
	}

	outFile << "ncols "        << esriheader.ncols     << endl;
	outFile << "nrows "        << esriheader.nrows     << endl;
	outFile << "xllcorner "    << esriheader.xllcorner << endl;
	outFile << "yllcorner "    << esriheader.yllcorner << endl;
	outFile << "cellsize "     << esriheader.cellsize  << endl;
	outFile << "NODATA_value " << esriheader.NODATA    << endl;

	write_func(&outFile);
}

void dataSaver::write_func(ofStream* outFile)
{
	deque< deque <double> >* cur_lines = new deque< deque < double> >;
	int count = 0;
	int i;
	
	curr_lines -> push_back(deque<double> (esriheader.ncols, esriheader.NODATA));

	for(i = 0; i < 2; i++)
	{
		boost::mutex::scoped_lock lock(buffer_lock);
		while(buffer.size() == 0)
		{
			buffer_available.wait(buffer_lock);
		}

		curr_lines -> push_back(buffer.front());
		buffer.pop_front();
		buffer_available.notify_one();
		buffer_lock.unlock();

		count++;
	}
	
	for(i = 0; i < esriheader.ncols; i++)
	{
			*outFile << 
	}
	*outFile << endl;
	
	while(count < esriheader.nrows)
	{
		curr_lines -> pop_front();
		
		boost::mutex::scoped_lock lock(buffer_lock);
		while(buffer.size() == 0)
		{
			buffer_available.wait(buffer_lock);
		}
		curr_lines -> puch_back(buffer.front());
		buffer.pop_front();
		buffer_available.notify_one();
		buffer_lock.unlock();

		count++;
		for(i = 0; i < esriheaderncols; i++)
		{
			*outFile << 
		}
		*outFile << endl;
	}
	
	curr_lines -> pop_front();
	curr_lines -> push_back(deque<double> (esriheader.ncols, esriheader.NODATA));
	
	for(i = 0; i < esriheader.ncols; i++)
	{
		*outFile <<
	}
	*outFile << endl;
}

