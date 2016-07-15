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

/*
    Constructor for dataSaver
    parameters: 
        _fileName - path to asc file to be used as output
        _loadBuffer - pointer to shared buffer to receive data from    
        _load_buffer_available - boost condition to notify when load buffer is available
        _load_buffer_lock - lock on shared buffer to ensure thread-safe access
        _header - esriHeader object with header infotmation to write at beginning of file
        _functions - vector of function names to calculate
        _outBuffers - vector of out buffers to pass calculated data to
        _buffer_available_list - vector of boost conditions for outBuffers
        _buffer_lock_list - vector of boost locks for outBuffers
*/
serialCalc::serialCalc(std::deque< std::deque <double> >* _loadBuffer, std::vector< std::string >* _functions, esriHeader* _header,
                       boost::condition_variable_any* _load_buffer_available, boost::mutex* _load_buffer_lock, 
                       std::vector< std::deque< std::deque <double> >* >* _outBuffers, std::vector< boost::condition_variable_any* >* _buffer_available_list,
                       std::vector< boost::mutex* >* _buffer_lock_list){

    this -> functions = _functions;

    this -> input_buffer = _loadBuffer;
    this -> input_buffer_available = _load_buffer_available;
    this -> input_buffer_lock = _load_buffer_lock;

    this -> outBuffers = _outBuffers;
    this -> buffer_available_list = _buffer_available_list;
    this -> buffer_lock_list = _buffer_lock_list;
    this -> header_info = _header;
}

/*
    Deconstructor, currently does nothing special
*/
serialCalc::~serialCalc(){

}

/*
    Starts everything object needs to do
*/
void serialCalc::run(){
    run_func();
}

/*
    Determines which GIS function to compute and calls that function
*/
double serialCalc::calculate(std::deque< std::deque <double> >* cur_lines, int i, std::string function){
    switch(function[0]){
        case 's':
            return calc_slope(cur_lines, i);
        break;
        case 'h':
            return calc_hillshade(cur_lines, i);
        break;
        case 'a':
            return calc_aspect(cur_lines, i);
        break;
        default:
            std::cout << "Unsupported function type" << std::endl;
            return 0;
        break;
    }
}

/*
    saves 3 deques at a time to calculat slope of middle row
    passes that calculated row into output file
*/
void serialCalc::run_func(){
    std::deque< std::deque <double> >* cur_lines = new std::deque< std::deque <double> >;
    int count=0;
    int i;
    std::deque<double> temp;

    //First push back NODATA row for calculating sloep of first row
    cur_lines->push_back(std::deque<double> (header_info -> ncols, header_info -> NODATA));

    //Next, grab first two rows of data
    for(i=0; i<2; i++){
        //////////////////////LOCK/////////////////////////
        boost::mutex::scoped_lock lock(*input_buffer_lock);
        while(input_buffer -> size() == 0){
            input_buffer_available -> wait(*input_buffer_lock);
        }   
        //DONT pop anything from cur_lines yet, need to fill with three rows.
        cur_lines->push_back(input_buffer -> front());
        input_buffer -> pop_front();
        input_buffer_available -> notify_one();
        ////////////////////UNLOCK/////////////////////////
        count++;
    }
    //std::cout << "grabbed first two data lines" << std::endl;
    for(unsigned q = 0; q < functions -> size(); ++q){
        //Calculate and write out first row

        for(i=0; i<header_info -> ncols; i++){
            temp.push_back(calculate(cur_lines, i, functions -> at(q)));
        }
        boost::condition_variable_any* output_buffer_available = buffer_available_list -> at(q);
        boost::mutex* output_buffer_lock = buffer_lock_list -> at(q);
        std::deque< std::deque <double> >* output_buffer =  outBuffers -> at(q);
        //////////////////////LOCK/////////////////////////
        // send calculated line into output buffer  ///////
        boost::mutex::scoped_lock lock(*output_buffer_lock);
        while(output_buffer -> size() == MAX_BUFF_SIZE){
            output_buffer_available -> wait(*output_buffer_lock);
        }
        output_buffer -> push_back(temp);
        output_buffer_available -> notify_one();
    }
    temp.clear();
    ////////////////////UNLOCK/////////////////////////

    //Enter main while loop
    while (count < header_info -> nrows){
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
        } while(false);
        ////////////////////UNLOCK/////////////////////////
        count++;

        for(unsigned q = 0; q < functions -> size(); ++q){
            for(i=0; i<header_info -> ncols; i++){
                temp.push_back(calculate(cur_lines, i, functions -> at(q)));
            }
            boost::condition_variable_any* output_buffer_available = buffer_available_list -> at(q);
            boost::mutex* output_buffer_lock = buffer_lock_list -> at(q);
            std::deque< std::deque <double> >* output_buffer =  outBuffers -> at(q);
            //////////////////////LOCK/////////////////////////
            // send calculated line into output buffer  ///////
            boost::mutex::scoped_lock lock(*output_buffer_lock);
            while(output_buffer -> size() == MAX_BUFF_SIZE){
                output_buffer_available -> wait(*output_buffer_lock);
            }
            output_buffer -> push_back(temp);
            output_buffer_available -> notify_one();
        }
        ////////////////////UNLOCK/////////////////////////
        temp.clear();
    }
    //Push back another NODATA row to calculate the last row with.
    cur_lines->pop_front();
    cur_lines->push_back(std::deque<double> (header_info -> ncols, header_info -> NODATA));
    //Calculate and write out last row
    for(unsigned q = 0; q < functions -> size(); ++q){
        for(i=0; i < header_info -> ncols; i++){
            temp.push_back(calculate(cur_lines, i, functions -> at(q)));
        }
        boost::condition_variable_any* output_buffer_available = buffer_available_list -> at(q);
        boost::mutex* output_buffer_lock = buffer_lock_list -> at(q);
        std::deque< std::deque <double> >* output_buffer =  outBuffers -> at(q);
        //////////////////////LOCK/////////////////////////
        // send calculated line into output buffer  ///////
        boost::mutex::scoped_lock lock(*output_buffer_lock);
        while(output_buffer -> size() == MAX_BUFF_SIZE){
            output_buffer_available -> wait(*output_buffer_lock);
        }
        output_buffer -> push_back(temp);
        output_buffer_available -> notify_one();
        ////////////////////UNLOCK/////////////////////////
        temp.clear();
    }
    delete cur_lines;
}

/* calc_slope - calculates the slope for a single cell in a raster
 *              file. cur_lines includes the input row of the cell, and
 *              the rows above and below that cell. col is the column
 *              of the cell we're calculating the slope for.
 */
double serialCalc::calc_slope(std::deque< std::deque <double> >* cur_lines, int col) {
        if (cur_lines->at(1)[col] == header_info -> NODATA){
            return header_info -> NODATA;
        }

        double nbhd[9];//'neighborhood' of current cell
        int k=0;
        
        for (int i=0; i<3; i++){
            for (int j=-1; j<2; j++){
                if ((col+j < 0) or (col+j > header_info -> ncols)){
                    nbhd[k] = header_info -> NODATA;
                }
                else{
                    nbhd[k] = cur_lines->at(i)[col+j];
                }
                k++;
            }
        }

        double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*header_info -> cellsize);
        double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*header_info -> cellsize);

        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
}

/*
    calculates aspect for a single cell at col using a 3x3 kernel
*/
double serialCalc::calc_aspect(std::deque< std::deque <double> >* cur_lines, int col){
    if (cur_lines->at(1)[col] == header_info -> NODATA){
        return header_info -> NODATA;
    }

    double nbhd[9];//'neighborhood' of current cell
    int k=0;
        
    for (int i=0; i<3; i++){
        for (int j=-1; j<2; j++){
            if ((col+j < 0) or (col+j > header_info -> ncols)){
                nbhd[k] = header_info -> NODATA;
            }
             else{
                nbhd[k] = cur_lines->at(i)[col+j];
            }
            k++;
        }
    }

    double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*header_info -> cellsize);
    double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*header_info -> cellsize);
    double aspect = 57.29578 * (atan2(dz_dy, -(dz_dx)));
        if(dz_dx == header_info -> NODATA || dz_dy == header_info -> NODATA || (dz_dx == 0.0 && dz_dy == 0.0)){
            return header_info -> NODATA;
        } else{
            if(aspect > 90.0){
                aspect = 360.0 - aspect + 90.0;
            } else {
                aspect = 90.0 - aspect;
            }
            aspect = aspect * (M_PI / 180.0);
            return aspect;
        }
}

/*
    calculates hillshade for a single cell at col using a 3x3 kernel
*/
double serialCalc::calc_hillshade(std::deque< std::deque <double> >* cur_lines, int col){
    if (cur_lines->at(1)[col] == header_info -> NODATA){
        return header_info -> NODATA;
    }
    double nbhd[9];//'neighborhood' of current cell
    int k=0;
    for (int i=0; i<3; i++){
        for (int j=-1; j<2; j++){
            if ((col+j < 0) or (col+j > header_info -> ncols)){
                nbhd[k] = header_info -> NODATA;
            }
            else{
                nbhd[k] = cur_lines->at(i)[col+j];
            }
            k++;
        }
    }
    double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*header_info -> cellsize);
    double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*header_info -> cellsize);

    double slp = calc_slope(cur_lines, col);
    double asp = hillshade_aspect(dz_dx, dz_dy);

                        /* calc zenith */
    double altitude = 45;
	double zenith_deg = 90 - altitude;
	double zenith_rad = zenith_deg * (M_PI / 180.0);
	
                        /* calc azimuth */
	double azimuth = 315;
	double azimuth_math = (360 - azimuth + 90);
	if(azimuth_math >= 360.0){
	    azimuth_math = azimuth_math - 360;
    }	
    double azimuth_rad = (azimuth_math * M_PI / 180.0);

    double hs = 255.0 * ( ( cos(zenith_rad) * cos(slp) ) + ( sin(zenith_rad) * sin(slp) * cos(azimuth_rad - asp) ) );

	if(hs < 0){
	    return 0;
    } else {
        return hs;
    }
}

/*
    calculates aspect needed for hillshade
*/
double serialCalc::hillshade_aspect(double dz_dx, double dz_dy){
    double aspect;
    if(dz_dx != 0){
        aspect = atan2(dz_dy, -(dz_dx));
        if(aspect < 0){
            aspect = ((2 * M_PI) + aspect);
        }
    } else if(dz_dx == 0){
        if(dz_dy > 0){
            aspect = (M_PI / 2);
        }else if(dz_dy < 0){
            aspect = ((2 * M_PI) - (M_PI / 2));
        }else{
            aspect = atan2(dz_dy, -(dz_dx));
        }
    }
    return aspect;
}
