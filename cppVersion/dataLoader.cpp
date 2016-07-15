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
#include "dataLoader.h"

const static int MAX_BUF_SIZE = 70000;

/*
    Constructor for dataLoader  
    parameters: 
        _fileName - path to asc file to be used as input
        _buffer - pointer to shared buffer to pass data into    
        _buffer_available - boost condition to notify when buffer is available
        _buffer_lock - lock on shared buffer to ensure thread-safe access
*/
dataLoader::dataLoader(std::string _fileName, std::deque<std::deque <double> >* _buffer, 
                        boost::condition_variable_any* _buffer_available, boost::mutex* _buffer_lock){
    this -> fileName = _fileName;
    this -> buffer = _buffer;
    this -> buffer_available = _buffer_available;
    this -> buffer_lock = _buffer_lock;
    openFile();
    readHeader();
}

/*
    Simple destructor for dataLoader
    just closes input file
*/
dataLoader::~dataLoader(){
    inFile.close();
}

/*
    Opens file, reads header, and starts reading input file
*/
void dataLoader::run(){
    std::cout<< "Starting to read" << std::endl;
    readFile();
}

/*
    return header info as a string
*/
std::string dataLoader::getHeader(){
    std::stringstream s;
    s << "ncols = " << ncols << "\n";
    s << "nrows = " << nrows << "\n";
    s << "cellsize = " << cellsize << "\n";
    s << "NODATA = " << nodata << "\n";
    s << "xllcorner = " << xllcorner << "\n";
    s << "yllcorner = " << yllcorner << "\n";
    return s.str();
}

/*
    Opens fileName given to constructor and puts ifstream object in inFile
*/
void dataLoader::openFile(){
    inFile.open(fileName.c_str());
    if(!inFile.is_open()) {
        std::cerr << "File failed to open" << "\n";
        exit(1);
    }
}

/*
    Reads inFile one line at a time and passes the double version of 
    the asc line into its shaed buffer. Assumes the file pointer    
    is at the first line of data 
*/
void dataLoader::readFile(){
    std::string line;
    int i = 0;
    std::deque< double > row;

    //Read file line by line
    while(std::getline(inFile, line)) {
        //removing whitespace in the begining of the lines
        line.erase(line.begin());
        std::istringstream ss(line);
        std::string x;
        //Copy each token from a line into deque
        while(std::getline(ss, x, ' ')) {
            row.push_back(atof(x.c_str()));
        }
        ///////////////LOCK////////////////////
        boost::mutex::scoped_lock lock(*buffer_lock);
        while(buffer -> size() == MAX_BUF_SIZE){
            buffer_available -> wait(*buffer_lock);
        }
        buffer -> push_back(row);
        buffer_available -> notify_one();
        row.clear();
        ++i;
        /////////////UNLOCK///////////////////
   }

    if(inFile.eof()){
        std::cout << "EOF reaced" << std::endl;
    }
    return;
}

/*
    Reads the first 6 lines of the header to get raster information
    store them in instance vars. Requires file to already be opened
    Advances the file pointer to the first line of actual data
*/
void dataLoader::readHeader(){
    int temp, count;
	temp = count = 0;
    std::string header;
	//reading lines of header and saving keyvalues into global variables
	while (count < 6){
		std::getline(inFile, header);
		if(count == 0){
            while(header[temp++] != ' '){}
			ncols = atol(header.substr(temp+1, header.length()-1).c_str());

		} else if(count == 1){
            while(header[temp++] != ' '){}
            nrows = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 2){
            while(header[temp++] != ' '){}
            xllcorner = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 3){
            while(header[temp++] != ' '){}
            yllcorner = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 4){
            while(header[temp++] != ' '){}
            cellsize = atof(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 5){
            while(header[temp++] != ' '){}
            nodata = atof(header.substr(temp+1, header.length()-1).c_str());
        }
    temp = 0;
    ++count;
	}
}

