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
#include "dataSaver.h"
#include "esriHeader.h"

dataSaver::dataSaver(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available,
		 boost::mutex *buffer_lock, esriHeader* header)
{
	this -> fileName = fileName;
	this -> buffer = buffer;
	this -> buffer_available = buffer_available;
	this -> buffer_lock = buffer_lock;
    this -> header = header;
	//openFile();
}

/*
    Starts everything object needs to do
*/
void dataSaver::run(){
    openFile();
    write_func();
}

void dataSaver::openFile()
{
	//open outputfile
	outFile.open(fileName.c_str());
	if (!outFile.is_open())
	{
		std::cerr << "File failed to open" << std::endl;
		exit(1);	
	}
	//write header info to outputfile
	outFile << "ncols "        << header -> ncols     << '\n';
	outFile << "nrows "        << header -> nrows     << '\n';
	outFile << "xllcorner "    << header -> xllcorner << '\n';
	outFile << "yllcorner "    << header -> yllcorner << '\n';
	outFile << "cellsize "     << header -> cellsize  << '\n';
	outFile << "NODATA_value " << header -> NODATA    << '\n';
}

//write data to the output file
void dataSaver::write_func()
{
    std::deque< std::deque <double> >* cur_lines = new std::deque< std::deque < double> >;
    int count = 0;
    unsigned i;
    //enter main while loop
    while(count < header -> nrows)
    {
        //LOCK
        boost::unique_lock<boost::mutex> lock(*buffer_lock);
        //boost::mutex::_lock lock(*buffer_lock);
        while(buffer -> size() == 0){
            buffer_available -> wait(*buffer_lock);            
        }
        // grab as many rows as are in buffer
        //for(i = 0; i < buffer -> size(); ++i){
        cur_lines -> push_back(buffer -> front());
        buffer -> pop_front();
        ++count;
        //}
        buffer_available -> notify_one();
        buffer_lock -> unlock();
        //UNLOCK
        for(i = 0; i < cur_lines -> size(); ++i){
            for(int q = 0; q < header -> ncols; ++q)
            {
                outFile << cur_lines -> front().at(q) << ' '; //need to figure out what command to call here to write to outfile
            }
            outFile << '\n';
            cur_lines -> pop_front();
        }
    }
}

