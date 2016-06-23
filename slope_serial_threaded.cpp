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

using namespace std;

//declaring global variables for # of columns, rows, cell size, and no data value
int NCOLS, NROWS;
double CELLSIZE, NODATA;

//Variables shared between threads
boost::condition_variable_any buffer_available;
boost::mutex buffer_lock;
deque<deque <double> > buffer;
const static int MAX_BUF_SIZE = 70000;

//function headers
void getHeader(ifstream* inFile, ofstream* outFile, string header[]);
void loadData(ifstream* inFile);
void calcFunc(ofstream* outFile);
double calc_slope(deque< deque <double> >* cur_lines, int col);



int main(int argc, char* argv[])
{
        if (argc != 3)
        {
                cerr << "Incorrect number of args" << endl;
                cout << "Correct usage: ./Slope <input filename> <output filename>" << endl;
                return 1;
        }
	
        string filename(argv[1]);
	string outputFile(argv[2]);
	string header[6];

        //Open files
        ifstream inFile;
        inFile.open(filename.c_str());
        if(!inFile.is_open())
        {
                cerr << "File failed to open" << endl;
                exit(1);
        }

	ofstream outFile;
	outFile.open(outputFile.c_str());
	if(!outFile.is_open())
	{
		cerr << "File failed to open" << endl;
		exit(1);
	}
	
        getHeader(&inFile, &outFile, header);

        //start threads
        boost::thread load_thread(loadData, &inFile);
        boost::thread calc_thread(calcFunc, &outFile);

        //wait for thread completion
        load_thread.join();
        calc_thread.join();

        inFile.close();
        outFile.close();

        return 0;
}

/* getHeader - Gathers header information from input file, and then
 *             writes that information to the output file since it will
 *             be identical.
 */
void getHeader(ifstream* inFile, ofstream* outFile, string header[])
{
	int count = 0;
	vector<string> keyValues;

	//reading lines of header and saving keyvalues into global variables
	while (count < 6)
	{
		getline(*inFile, header[count]);
		if(count == 0)
		{
			istringstream headLine(header[count]);			
			copy(istream_iterator<string>(headLine),
				istream_iterator<string>(),
				back_inserter(keyValues));
			NCOLS = atoi(keyValues[1].c_str());
			count++;
		}
		else if(count == 1)
                {
			istringstream headLine(header[count]);
                        copy(istream_iterator<string>(headLine),
                                istream_iterator<string>(),
                                back_inserter(keyValues));
                        NROWS = atoi(keyValues[3].c_str());
			count++;
                }
		else if(count == 4)
                {
			istringstream headLine(header[count]);
                        copy(istream_iterator<string>(headLine),
                                istream_iterator<string>(),
                                back_inserter(keyValues));
                        CELLSIZE = atof(keyValues[5].c_str());
			count++;
                }
		else if(count == 5)
                {
			istringstream headLine(header[count]);
                        copy(istream_iterator<string>(headLine),
                                istream_iterator<string>(),
                                back_inserter(keyValues));
                        NODATA = atof(keyValues[7].c_str());
			count++;
                }
		else
			count++;
	}

        //Write header to output file
	for (int i = 0; i < 6; i++)
	{
		*outFile << header[i] << '\n';
	}
}

/* loadData - Thread function which loads data into the shared buffer,
 *            line by line.
 */
void loadData(ifstream* inFile){
        string line;
        deque< double > row;

        //Read file line by line
        while(getline(*inFile, line))
        {
		//removing whitespace in the begining of the lines
                line.erase(line.begin());
		istringstream ss(line);
                string x;
                //Copy each token from a line into deque
                while(getline(ss, x, ' '))
		{
                        row.push_back(atof(x.c_str()));
          	}
                ///////////////LOCK////////////////////
                boost::mutex::scoped_lock lock(buffer_lock);
                while(buffer.size() == MAX_BUF_SIZE){
                    buffer_available.wait(buffer_lock);
                }
                buffer.push_back(row);
                buffer_available.notify_one();
                buffer_lock.unlock();
                /////////////UNLOCK///////////////////
                row.clear();
        }
}

/* calcFunc - Thread function which consumes data from the shared buffer,
 *            line by line. Also writes results of calculations to
 *            output file.
 */
void calcFunc(ofstream* outFile){
    deque< deque <double> >* cur_lines = new deque< deque <double> >;
    int count=0;
    int i;
    double cur_slope[NCOLS];

    //First push back NODATA row for calculating sloep of first row
    cur_lines->push_back(deque<double> (NCOLS, NODATA));
    //Next, grab first two rows of data
    for(i=0; i<2; i++){
        //////////////////////LOCK/////////////////////////
        boost::mutex::scoped_lock lock(buffer_lock);
        while(buffer.size() == 0){
            buffer_available.wait(buffer_lock);
        }
        //DONT pop anything from cur_lines yet, need to fill with three rows.
        cur_lines->push_back(buffer.front());
        buffer.pop_front();
        buffer_available.notify_one();
        buffer_lock.unlock();
        ////////////////////UNLOCK/////////////////////////
        count++;
    }
    //Calculate and write out first row
    for(i=0; i<NCOLS; i++){
        *outFile << calc_slope(cur_lines, i) << " ";
    }
    *outFile << "\n";

    //Enter main while loop
    while (count < NROWS){
        cur_lines->pop_front();
        //////////////////////LOCK/////////////////////////
        boost::mutex::scoped_lock lock(buffer_lock);
        while(buffer.size() == 0){
            buffer_available.wait(buffer_lock);
        }
        cur_lines->push_back(buffer.front());
        buffer.pop_front();
        buffer_available.notify_one();
        buffer_lock.unlock();
        ////////////////////UNLOCK/////////////////////////
        count++;
        for(i=0; i<NCOLS; i++){
            *outFile << calc_slope(cur_lines, i) << " ";
        }
        *outFile << "\n";
    }


    //Push back another NODATA row to calculate the last row with.
    cur_lines->pop_front();
    cur_lines->push_back(deque<double> (NCOLS, NODATA));
    //Calculate and write out last row
    for(i=0; i<NCOLS; i++){
        *outFile << calc_slope(cur_lines, i) << " ";
    }
    *outFile << "\n";

    delete cur_lines;
}

/* calc_slope - calculates the slope for a single cell in a raster
 *              file. cur_lines includes the input row of the cell, and
 *              the rows above and below that cell. col is the column
 *              of the cell we're calculating the slope for.
 */
double calc_slope(deque< deque <double> >* cur_lines, int col)
{
        if (cur_lines->at(1)[col] == NODATA){
            return NODATA;
        }

        double nbhd[9];//'neighborhood' of current cell
        int k=0;
        
        for (int i=0; i<3; i++){
            for (int j=-1; j<2; j++){
                if ((col+j < 0) or (col+j > NCOLS)){
                    nbhd[k] = NODATA;
                }
                else{
                    nbhd[k] = cur_lines->at(i)[col+j];
                }
                k++;
            }
        }

        double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*CELLSIZE);
        double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*CELLSIZE);

        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
}
