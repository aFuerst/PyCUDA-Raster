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
#include "serialCalc.h"
#include "esriHeader.h"


serialCalc::serialCalc(std::deque<std::deque <double> > *input_buffer, boost::condition_variable_any *input_buffer_available, boost::mutex *input_buffer_lock, 
               std::deque<std::deque <double> > *output_buffer, boost::condition_variable_any *output_buffer_available, boost::mutex *output_buffer_lock, esriHeader header_info){

    this -> input_buffer = input_buffer;
    this -> input_buffer_available = input_buffer_available;
    this -> input_buffer_lock = input_buffer_lock;
    this -> output_buffer = output_buffer;
    this -> output_buffer_available = output_buffer_available;
    this -> output_buffer_lock = output_buffer_lock;
    this -> header_info;
}

/*
    saves 3 deques at a time to calculat slope of middle row
    passes that calculated row into output file


*/
void serialCalc::run_func(){
    std::deque< std::deque <double> >* cur_lines = new std::deque< std::deque <double> >;
    int count=0;
    int i;
    double cur_slope[header_info.ncols];

    //First push back NODATA row for calculating sloep of first row
    cur_lines->push_back(std::deque<double> (header_info.ncols, header_info.NODATA));

    //Next, grab first two rows of data
    for(i=0; i<2; i++){
        //////////////////////LOCK/////////////////////////
        do{
            boost::mutex::scoped_lock lock(*input_buffer_lock);
            while(input_buffer -> size() == 0){
                input_buffer_available -> wait(*input_buffer_lock);
            }
            //DONT pop anything from cur_lines yet, need to fill with three rows.
            cur_lines->push_back(input_buffer -> front());
            input_buffer -> pop_front();
            input_buffer_available -> notify_one();
            input_buffer_lock -> unlock();
        } while(false);
        ////////////////////UNLOCK/////////////////////////
        count++;
    }

    //Calculate and write out first row
    std::deque<double> temp;
    for(i=0; i<header_info.ncols; i++){
        temp.push_back(calc_slope(cur_lines, i));
    }
    temp.clear();
    //////////////////////LOCK/////////////////////////
    // send calculated line into output buffer  ///////
    do{
        boost::mutex::scoped_lock lock(*output_buffer_lock);
        while(output_buffer -> size() == 0){
            output_buffer_available -> wait(*output_buffer_lock);
        }
        output_buffer -> push_back(temp);
        output_buffer_available -> notify_one();
        output_buffer_lock -> unlock();
    } while(false);
    ////////////////////UNLOCK/////////////////////////

    //Enter main while loop
    while (count < header_info.nrows){
        cur_lines->pop_front();
        //////////////////////LOCK/////////////////////////
        // get new line
        do{
            boost::mutex::scoped_lock lock(*input_buffer_lock);
            while(input_buffer -> size() == 0){
                input_buffer_available -> wait(*input_buffer_lock);
            }
            cur_lines->push_back(input_buffer -> front());
            input_buffer -> pop_front();
            input_buffer_available -> notify_one();
            input_buffer_lock -> unlock();
        } while(false);
        ////////////////////UNLOCK/////////////////////////
        count++;
        for(i=0; i < header_info.ncols; i++){
            temp.push_back(calc_slope(cur_lines, i));
        }
        //////////////////////LOCK/////////////////////////
        // send calculated line into output buffer  ///////
        do{
            boost::mutex::scoped_lock lock(*output_buffer_lock);
            while(output_buffer -> size() == 0){
                output_buffer_available -> wait(*output_buffer_lock);
            }
            output_buffer -> push_back(temp);
            output_buffer_available -> notify_one();
            output_buffer_lock -> unlock();
        } while(false);
        ////////////////////UNLOCK/////////////////////////
        temp.clear();
    }

    //Push back another NODATA row to calculate the last row with.
    cur_lines->pop_front();
    cur_lines->push_back(std::deque<double> (header_info.ncols, header_info.NODATA));
    //Calculate and write out last row
    for(i=0; i < header_info.ncols; i++){
        temp.push_back(calc_slope(cur_lines, i));
    }
    //////////////////////LOCK/////////////////////////
    // send calculated line into output buffer  ///////
    boost::mutex::scoped_lock lock(*output_buffer_lock);
    while(output_buffer -> size() == 0){
        output_buffer_available -> wait(*output_buffer_lock);
    }
    output_buffer -> push_back(temp);
    output_buffer_available -> notify_one();
    output_buffer_lock -> unlock();
    ////////////////////UNLOCK/////////////////////////
    temp.clear();
    delete cur_lines;
}

/* calc_slope - calculates the slope for a single cell in a raster
 *              file. cur_lines includes the input row of the cell, and
 *              the rows above and below that cell. col is the column
 *              of the cell we're calculating the slope for.
 */
double serialCalc::calc_slope(std::deque< std::deque <double> >* cur_lines, int col) {
        if (cur_lines->at(1)[col] == header_info.NODATA){
            return header_info.NODATA;
        }

        double nbhd[9];//'neighborhood' of current cell
        int k=0;
        
        for (int i=0; i<3; i++){
            for (int j=-1; j<2; j++){
                if ((col+j < 0) or (col+j > header_info.ncols)){
                    nbhd[k] = header_info.NODATA;
                }
                else{
                    nbhd[k] = cur_lines->at(i)[col+j];
                }
                k++;
            }
        }

        double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*header_info.cellsize);
        double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*header_info.cellsize);

        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
}
