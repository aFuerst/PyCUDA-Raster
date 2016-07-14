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

    while (count < 6){
        std::getline(inFile, *header);
        if(count == 0){
            while(header->at(temp++) != ' '){}
            toReturn.ncols = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 1){
            while(header->at(temp++) != ' '){}
            toReturn.nrows = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 2){
            while(header->at(temp++) != ' '){}
            toReturn.xllcorner = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 3){
            while(header->at(temp++) != ' '){}
            toReturn.yllcorner = atol(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 4){
            while(header->at(temp++) != ' '){}
            toReturn.cellsize = atof(header->substr(temp+1, header->length()-1).c_str());

        } else if(count == 5){
            while(header->at(temp++) != ' '){}
            toReturn.NODATA = atof(header->substr(temp+1, header->length()-1).c_str());
        }
        temp = 0;
        ++count;
    }
    return toReturn;
}

void load_func(std::string inFile, std::deque< std::deque <double> >* loadBuffer, 
        boost::condition_variable_any* buffer_available, boost::mutex* buffer_lock){
    dataLoader loader(inFile, loadBuffer, buffer_available, buffer_lock);
    loader.run();

}

void calc_func(std::deque< std::deque <double> >* loadBuffer, std::vector< std::string >* functions, esriHeader* header,
        boost::condition_variable_any* load_buffer_available, boost::mutex* load_buffer_lock, 
        std::vector< std::deque< std::deque <double> >* >* outBuffers, std::vector< boost::condition_variable_any* >* buffer_available_list,
        std::vector< boost::mutex* >* buffer_lock_list){

    serialCalc calc(loadBuffer, functions, header, load_buffer_available, load_buffer_lock, outBuffers, buffer_available_list, buffer_lock_list);
    calc.run();

}

void save_func(std::string outFile ,std::deque< std::deque <double> >* saveBuffer, esriHeader* header, 
        boost::condition_variable_any* buffer_available, boost::mutex* buffer_lock){
    dataSaver save(outFile, saveBuffer, buffer_available, buffer_lock, header);
    save.run();

}

int main(int argc, char* argv[]){
    std::vector< std::string > outFiles;
    std::vector< std::string > functions;
    for (int i=2; i<argc; i+=2){
        outFiles.push_back(argv[i]);
        functions.push_back(argv[i+1]);
    }

    esriHeader header = getHeader(argv[1]);

    /*for (int i=0; i<outFiles.size(); i++){
        std::cout << outFiles.at(i) << std::endl;
        std::cout << functions.at(i) << std::endl;
    }*/

    boost::thread_group saverThreads;

    // locks for load buffer
    boost::condition_variable_any* load_buffer_available = new boost::condition_variable_any;
    boost::mutex* load_buffer_lock = new boost::mutex;

    std::deque< std::deque <double> >* loadBuffer = new std::deque< std::deque <double> >;
    boost::thread loadThread(load_func, argv[1], loadBuffer, load_buffer_available, load_buffer_lock);


    // vectors to hold variable num of output buffers and locks
    std::vector< std::deque< std::deque <double> >* > outBuffers;
    std::vector< boost::condition_variable_any* > buffer_available_list;
    std::vector< boost::mutex* > buffer_lock_list;
    std::vector< boost::thread* > thread_list;
    

    for (unsigned i=0; i<outFiles.size(); i++){
        // create all output buffers and locks
        outBuffers.push_back(new std::deque< std::deque <double> >);
        buffer_available_list.push_back(new boost::condition_variable_any);
        buffer_lock_list.push_back(new boost::mutex);
        boost::thread* saveThread = new boost::thread(save_func, outFiles[i], outBuffers.at(i), &header, buffer_available_list.at(i), buffer_lock_list.at(i));
        saverThreads.add_thread(saveThread);
    }

    //boost::thread saveThread(save_func, outFiles[0], outBuffers.at(0), &header, buffer_available_list.at(0), buffer_lock_list.at(0));
    boost::thread calcThread(calc_func, loadBuffer, &functions, &header, load_buffer_available, load_buffer_lock, &outBuffers, &buffer_available_list, &buffer_lock_list);
    
    std::cout << "joining threads" << std::endl;

    loadThread.join();
    std::cout << "load thread joined" << std::endl;
    //saveThread.join();
    calcThread.join();
    std::cout << "calc thread joined" << std::endl;

    saverThreads.join_all();

    std::cout << "save threads joined" << std::endl;

    delete load_buffer_available;
    delete load_buffer_lock;
    delete loadBuffer;
    outBuffers.clear();
    buffer_available_list.clear();
    buffer_lock_list.clear();

    std::cout << "END\n";
    return 0;
}

