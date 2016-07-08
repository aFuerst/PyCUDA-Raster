#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <boost/thread.hpp>
#include "dataLoader.h"
#include "dataSaver.h"
#include "serialCalc.h"

int main(int argc, char* argv[]){
    std::vector< std::string > outFiles;
    std::vector< std::string > functions;
    for (int i=2; i<argc; i+=2){
        outFiles.push_back(argv[i]);
        functions.push_back(argv[i+1]);
    }

    for (int i=0; i<outFiles.size(); i++){
        std::cout << outFiles.at(i) << std::endl;
        std::cout << functions.at(i) << std::endl;
    }

    std::deque< std::deque <double> >* loadBuffer = new std::deque< std::deque <double> >;
    std::deque< std::deque <double> >* outBuffer = new std::deque< std::deque <double> >;

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
