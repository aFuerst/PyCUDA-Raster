/**************************************************
 * Note, just prints to stdout. Pipe to a file if desired.
 **************************************************/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iterator>

//declaring global variables for # of columns, rows, cell size, and no data value
int NCOLS, NROWS;
float CELLSIZE, NODATA;

using namespace std;

//function headers
float** loadFile(string filename, string header[]);
float** calcSlope(float** grid);
float cellSlope(float** grid, int row, int col);
void output(string outputFile,string header[], float** grid);

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

        float** grid = loadFile(filename, header);
        float** slope_grid = calcSlope(grid);
	output(outputFile, header, slope_grid);

	
        for (int i=0; i<NROWS; i++)
        {
                delete[] grid[i];
                delete[] slope_grid[i];
        }
        delete[] grid;
        delete[] slope_grid;

        return 0;
}

float** loadFile(string filename, string header[])
{
        //Open file
        ifstream inFile;
        inFile.open(filename.c_str());
        if(!inFile.is_open())
        {
                cerr << "File failed to open" << endl;
                exit(1);
        }
        
	int count = 0;
	vector<string> keyValues;

	//reading lines of header and saving keyvalues into global variables
	while (count < 6)
	{
		getline(inFile, header[count]);
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
}

//outputing header and calculated slope data to a file
void output(string outputFile, string header[], float** grid)
{
	ofstream outFile;
	outFile.open(outputFile.c_str());
	if(!outFile.is_open())
	{
		cerr << "File failed to open" << endl;
		exit(1);
	}
	
	int i, j;
	
	for (i = 0; i < 6; i++)
	{
		outFile << header[i] << '\n';
	}

	for (i = 0; i < NROWS; i++)
	{
		for (j = 0; j < NCOLS; j++)
		{
			outFile << grid[i][j] << " ";
		}
		outFile << endl;
	}

	outFile.close();
}

float** calcSlope(float** grid)
{
        float** slope_grid = new float* [NROWS];

        for (int row=0; row<NROWS; row++)
        {
                slope_grid[row] = new float [NCOLS];
                for (int col=0; col<NCOLS; col++)
                {
                        slope_grid[row][col] = cellSlope(grid, row, col);
                }
        }
        return slope_grid;
}

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
