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

class dataLoader{

    public:
        dataLoader(string fileName);
        string getHeader(void);

    private:
        void readHeader();
        string readLine();
        void openFile();

        long long ncols;
        long long nrows;
        double cellsize;
        double nodata;
        long long xllcorner;
        long long yllcorner;

        string header;
        string fileName;
}

dataLoader::dataLoader(string fileName){
    this.fileName = fileName;
}

dataLoader::readHeader(){
	int count = 0;
    string header;
	//reading lines of header and saving keyvalues into global variables
	while (count < 6){
		getline(*inFile, header);
		if(count == 0){
            temp = 0;
            while(header[temp] != '='){
                
            }
			ncols = atoi(keyValues[1].c_str());

		} else if(count == 1){
			istringstream headLine(header[count]);
            copy(istream_iterator<string>(headLine), istream_iterator<string>(), back_inserter(keyValues));
            nrows = atoi(keyValues[3].c_str());

        } else if(count == 2){
			istringstream headLine(header[count]);
            copy(istream_iterator<string>(headLine), istream_iterator<string>(), back_inserter(keyValues));
            xllcorner = atoi(keyValues[5].c_str());

        } else if(count == 3){
		    istringstream headLine(header[count]);
            copy(istream_iterator<string>(headLine), istream_iterator<string>(), back_inserter(keyValues));
            yllcorner = atoi(keyValues[7].c_str());

        } else if(count == 4){
			istringstream headLine(header[count]);
            copy(istream_iterator<string>(headLine), istream_iterator<string>(), back_inserter(keyValues));
            cellsize = atof(keyValues[9].c_str());

        } else if(count == 5){
			istringstream headLine(header[count]);
            copy(istream_iterator<string>(headLine), istream_iterator<string>(), back_inserter(keyValues));
            nodata = atof(keyValues[11].c_str());
        }
     ++count;
	}
}

