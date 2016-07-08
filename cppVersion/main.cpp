#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <boost/thread.hpp>
#include "dataLoader.h"
#include "dataSaver.h"
#include "serialCalc.h"
#include "esriHeader.h"

esriHeader getHeader(std::string fileName){
    std::ifstream inFile;
    inFile.open(fileName.c_str());
    if(!inFile.is_open()) {
        std::cerr << "File failed to open" << "\n";
        exit(1);
    }

    std::cout << "Opened file" << "\n";

    int temp, count;
    temp = count = 0;
    std::string* header = new std::string;

    esriHeader toReturn;

    std::cout << "Entering loop" << "\n";
    while (count < 6){
        std::getline(inFile, *header);
        if(count == 0){
            while(header->at(temp++) != '='){}
            toReturn.ncols = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 1){
            while(header->at(temp++) != '='){}
            toReturn.nrows = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 2){
            while(header->at(temp++) != '='){}
            toReturn.xllcorner = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 3){
            while(header->at(temp++) != '='){}
            toReturn.yllcorner = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 4){
            while(header->at(temp++) != '='){}
            toReturn.cellsize = atof(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 5){
            while(header->at(temp++) != '='){}
            toReturn.NODATA = atof(header->substr(temp+1, header->length()-1).c_str());
        }
        temp = 0;
        ++count;
    }
    std::cout << toReturn.ncols << "\n" << toReturn.nrows << "\n" << toReturn.cellsize;
    return toReturn;
}

int main(int argc, char* argv[]){
    std::vector< std::string > outFiles;
    std::vector< std::string > functions;
    for (int i=2; i<argc; i+=2){
        outFiles.push_back(argv[i]);
        functions.push_back(argv[i+1]);
    }

    std::cout << argv[1] << "\n";
    esriHeader header = getHeader(argv[1]);

    //for (int i=0; i<outFiles.size(); i++){
    //    std::cout << outFiles.at(i) << std::endl;
    //    std::cout << functions.at(i) << std::endl;
    //}

    //std::deque< std::deque <double> >* loadBuffer = new std::deque< std::deque <double> >;
    //boost::thread loadThread(argv[1], loadBuffer);

    //std::vector< std::deque< std::deque <double> >* > outBuffers;
    //for (int i=0; i<outFiles.size(); i++){
    //    outBuffers.push_back(new std::deque< std::deque <double> >);
    //}
    //boost::thread calcThread(loadBuffer);

    return 0;
}

void loadThread(){
    return;
}

void calcThread(){
    return;
}

void saveThread(){
    return;
}
