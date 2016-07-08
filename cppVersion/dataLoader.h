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
        dataLoader(std::string fileName);
        std::string getHeader(void);

    private:
        void readHeader();
        std::vector<double> readLine();
        void openFile();

        long long ncols;
        long long nrows;
        double cellsize;
        double nodata;
        long long xllcorner;
        long long yllcorner;

        std::string fileName;
        std::ifstream inFile;
};

dataLoader::dataLoader(std::string fileName){
    this -> fileName = fileName;
    openFile();
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
    std::string x;
    std::vector<double> row;
    while(std::getline(inFile, x, ' ')) {
        row.push_back(atof(x.c_str()));
    }
    return row;
}

void dataLoader::readHeader(){
    int temp, count;
	temp = count = 0;
    std::string header;
	//reading lines of header and saving keyvalues into global variables
	while (count < 6){
		std::getline(inFile, header);
		if(count == 0){
            while(header[temp++] != '='){}
			ncols = atol(header.substr(temp+1, header.length()-1).c_str());

		} else if(count == 1){
            while(header[temp++] != '='){}
            nrows = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 2){
            while(header[temp++] != '='){}
            xllcorner = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 3){
            while(header[temp++] != '='){}
            yllcorner = atol(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 4){
            while(header[temp++] != '='){}
            cellsize = atof(header.substr(temp+1, header.length()-1).c_str());

        } else if(count == 5){
            while(header[temp++] != '='){}
            nodata = atof(header.substr(temp+1, header.length()-1).c_str());
        }
    temp = 0;
    ++count;
	}
}

