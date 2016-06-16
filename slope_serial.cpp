/**************************************************
 * Note, just prints to stdout. Pipe to a file if desired.
 **************************************************/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>

//static const int NCOLS = 1051;
//static const int NROWS = 1407;
//static const float CELLSIZE = 10.0;
//static const float NODATA = -32767.0;

int NCOLS, NROWS;
float CELLSIZE, NODATA;
string header[6];

using namespace std;

float** loadFile(string filename);
float** calcSlope(float** grid);
float cellSlope(float** grid, int row, int col);

int main(int argc, char* argv[]){
        if (argc != 2){
                cerr << "Incorrect number of args" << endl;
                cout << "Correct usage: ./slope_serial <filename>" << endl;
                return 1;
        }

        string filename(argv[1]);
        int i, j;

        //TODO: Free grid at end of program
        float** grid = loadFile(filename);
        float** slope_grid = calcSlope(grid);
        for (i=0; i<NROWS; i++){
                for (int j=0; j<NCOLS; j++){
                        cout << slope_grid[i][j] << " ";
                }
                cout << endl;
        }

        for (i=0; i<NROWS; i++){
                delete[] grid[i];
                delete[] slope_grid[i];
        }
        delete[] grid;
        delete[] slope_grid;

        return 0;
}

float** loadFile(string filename){
        //Open file
        ifstream inFile;
        inFile.open(filename.c_str());
        if(!inFile.is_open()){
                cerr << "File failed to open" << endl;
                exit(1);
        }
        
        int count = 0;
        vector<string> splits;
        
        while (count < 6)
	{
		//read in header line one at a time
		getline(inFile, header[count]);
		//grab number of cols and make into int variable
		if (count == 0)
		{
			istringstream stream(header[count]);
			copy(istream_iterator<string>(stream),
				istream_iterator<string>(),
				back_inserter(splits));
			NCOLS = stoi(splits[1], nullptr,10);
			count++;
		}
		//grab number of rows and make into int variable
		else if (count == 1)
		{
			istringstream stream(header[count]);
			copy(istream_iterator<string>(stream),
				istream_iterator<string>(),
				back_inserter(splits));
			NROWS = stoi(splits[3], nullptr, 10);
			count++;
		}
		//grab cell size and maake into float variable
		else if (count == 4)
		{
			istringstream stream(header[count]);
			copy(istream_iterator<string>(stream),
				istream_iterator<string>(),
				back_inserter(splits));
			CELLSIZE = stof(splits[5]);
			count++;
		}
		//grab no data value and make into float variable
		else if (count == 5)
		{
			istringstream stream(header[count]);
			copy(istream_iterator<string>(stream),
				istream_iterator<string>(),
				back_inserter(splits));
			NODATA = stof(splits[7]);
			count++;
		}
		//go to next line in header
		else
		count++;
	}
	
        float** grid = new float* [NROWS];
        for(int k=0; k<NROWS; k++){
                grid[k] = new float [NCOLS];
        }
        
        string line;
        
        int i=0;
        int j=0;
        //Read file line by line
        while(getline(inFile, line)){
                istringstream ss(line);
                string x;
                //Copy each token from a line into a grid
                while(getline(ss, x, ' ')){
                        grid[i][j] = atof(x.c_str());
                        j++;
                }
                j=0;
                i++;
        }
        inFile.close();
        return grid;
}

float** calcSlope(float** grid){
        float** slope_grid = new float* [NROWS];

        for (int row=0; row<NROWS; row++){
                slope_grid[row] = new float [NCOLS];
                for (int col=0; col<NCOLS; col++){
                        slope_grid[row][col] = cellSlope(grid, row, col);
                }
        }
        return slope_grid;
}

float cellSlope(float** grid, int row, int col){
        float nbhd[9];
        int k = 0;
        for (int i=-1; i<2; i++){
                for (int j=-1; j<2; j++){
                        if ((row+i<=0) || (row+i>=NROWS) || (col+j<=0) || (col+j>=NCOLS)){
                                nbhd[k] = NODATA;
                        }
                        else{
                                nbhd[k] = grid[row+i][col+j];
                        }
                        k++;
                }
        }

        if (nbhd[4] == NODATA){
                return nbhd[4];
        }

        for (k=0; k<9; k++){
                if (nbhd[k] == NODATA){
                        nbhd[k] = nbhd[4];
                }
        }

        float dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*CELLSIZE);
        float dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*CELLSIZE);

        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
}
