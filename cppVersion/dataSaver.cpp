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

/*
    Constructor for dataSaver
    parameters: 
        _fileName - path to asc file to be used as output
        _buffer - pointer to shared buffer to pass data into    
        _buffer_available - boost condition to notify when buffer is available
        _buffer_lock - lock on shared buffer to ensure thread-safe access
        _header - esriHeader object with header infotmation to write at beginning of file
*/
dataSaver::dataSaver(std::string _fileName, std::deque<std::deque <double> > *_buffer, boost::condition_variable_any *_buffer_available,
		 boost::mutex *_buffer_lock, esriHeader* _header)
{
	this -> fileName = _fileName;
	this -> buffer = _buffer;
	this -> buffer_available = _buffer_available;
	this -> buffer_lock = _buffer_lock;
    this -> header = _header;
}

/*
    Deconstructor, closes output file
*/
dataSaver::~dataSaver(){
    outFile.close();
}

/*
    Opens file and begins write_func to print data to file
*/
void dataSaver::run(){
    openFile();
    write_func();
}

/*
    Opens file and writes header information to it
*/
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

/*
    Write data to the output file a single line at a time from its input buffer 
*/
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

