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
float CELLSIZE, NODATA;

//Variables shared between threads
//NOTE: May need to change to type confition_variable_any
boost::condition_variable_any buffer_available;
boost::mutex buffer_lock;
deque<deque <float> > buffer;
const static int MAX_BUF_SIZE = 10000;

//function headers
void getHeader(ifstream* inFile, ifstream* outFile, string header[]);

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

        inFile.close();
        outFile.close();

        return 0;
}

void getHeader(ifstream* inFile, ifstream* outFile, string header[])
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
	for (i = 0; i < 6; i++)
	{
		outFile << header[i] << '\n';
	}
}

void loadData(ifstream inFile){
        string line;
        deque< float > row;

        //Read file line by line
        while(getline(inFile, line))
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

void calcFunc(ifstream outFile){
    deque< deque <float> > cur_lines;
    int count=0;
    float cur_slope[NCOLS];

    while (count < NROWS){
        boost::mutex::scoped_lock lock(buffer_lock);
        while(buffer.size() == 0){
            buffer_available.wait(buffer_lock);
        }
        outFile << buffer.pop_front().pop_front() << "\n";
        count++;
    }
}

/*
	float** grid = new float* [NROWS];
        for(int k=0; k<NROWS; k++)
        {
                grid[k] = new float [NCOLS];
        }
        
        string line;
        
        int i=0;
        int j=0;
        //Read file line by line
        while(getline(inFile, line))
        {
		//removing whitespace in the begining of the lines
                line.erase(line.begin());
		istringstream ss(line);
                string x;
		char trash;
                //Copy each token from a line into a grid
                while(getline(ss, x, ' '))
		{
                        grid[i][j] = atof(x.c_str());
                        j++;
          	}
                j=0;
                i++;
        }
        inFile.close();
        return grid;
*/



/*
float cellSlope(float** grid, int row, int col)
{
        float nbhd[9];
        int k = 0;
        for (int i=-1; i<2; i++)
        {
                for (int j=-1; j<2; j++)
                {
                        if ((row+i<=0) || (row+i>=NROWS) || (col+j<=0) || (col+j>=NCOLS))
                        {
                                nbhd[k] = NODATA;
                        }
                        else
                        {
                                nbhd[k] = grid[row+i][col+j];
                        }
                        k++;
                }
        }

        if (nbhd[4] == NODATA)
        {
                return nbhd[4];
        }

        for (k=0; k<9; k++)
        {
                if (nbhd[k] == NODATA)
                {
                        nbhd[k] = nbhd[4];
                }
        }

        float dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*CELLSIZE);
        float dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*CELLSIZE);

        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
}
*/
