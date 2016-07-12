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

dataLoader::dataLoader(std::string fileName, std::deque<std::deque <double> > *buffer, boost::condition_variable_any *buffer_available, boost::mutex *buffer_lock){
    std::cout << "initializing loader\n";
    this -> fileName = fileName;
    this -> buffer = buffer;
    this -> buffer_available = buffer_available;
    this -> buffer_lock = buffer_lock;
    openFile();
}

/*
    Starts everything object needs to do
*/
void dataLoader::run(){
    std::cout<< "Starting to read\n";
    readLine();
}

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

void dataLoader::openFile(){
    inFile.open(fileName.c_str());
    if(!inFile.is_open()) {
        std::cerr << "File failed to open" << "\n";
        exit(1);
    }
}

std::vector<double> dataLoader::readLine(){
    std::cout<<"Alive\n";
    std::string line;
    int i = 0;
    std::deque< double > row;

    //Read file line by line
    while(std::getline(inFile, line)) {
        std::cout<<"Still alive\n";
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
        std::cout<<"STAYIN ALIVE\n";
        while(buffer -> size() == MAX_BUF_SIZE){
            buffer_available -> wait(*buffer_lock);
        }
        buffer -> push_back(row);
        buffer_available -> notify_one();
        buffer_lock -> unlock();
        /////////////UNLOCK///////////////////
        row.clear();
   }
/*
    while(std::getline(inFile, x, ' ') && i < ncols) {
        row.push_back(atof(x.c_str()));
        ++i;
    }
    return row;
*/
}

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

