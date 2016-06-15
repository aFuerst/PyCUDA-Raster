#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>

const int NCOLS = 1051;
const int NROWS = 1407;
const float CELLSIZE = 10.0;
const float NODATA = -32767.0;

using namespace std;

vector< vector <float> >* loadFile(string filename);

int main(int argc, char* argv[]){
        if (argc != 2){
                cerr << "Incorrect number of args" << endl;
                return 1;
        }

        string filename(argv[1]);

        //TODO: Free grid at end of program
        vector< vector <float> >* grid = loadFile(filename);

        delete grid;
        return 0;
}

vector< vector <float> >* loadFile(string filename){
        //Open file
        ifstream inFile;
        inFile.open(filename.c_str());
        if(!inFile.is_open()){
                cerr << "File failed to open" << endl;
                exit(1);
        }
        
        float* grid[] = new float*[NROWS];
        for(int k=0; k<NROWS; k++){
                grid[i] = new float[NCOLS];
        }
        
        
        //vector < vector < float > >* toReturn = new vector < vector <float> >;
        //vector < float > temp;
        string line;
        
        int i=0;
        int j=0;
        //Read file line by line
        while(getline(inFile, line)){
                istringstream ss(line);
                string x;
                //Copy each token from a line into a vector
                while(getline(ss, x, ' ')){
                        //temp.push_back(atof(x.c_str()));
                }
                //Add vector to grid
                //toReturn->push_back(temp);
                //temp.clear();
        }
        //for (int i=0; i<1000; i++){
        //        for (int j=0; j<1000; j++){
        //                cout << toReturn->at(i)[j] << endl;
        //        }
        //}
        inFile.close();
        //return toReturn;
}

vector< vector <float> >* calcSlope(vector< vector <float> >* grid, ncols, nrows, cellsize, NODATA){
        vector< vector <float> >* slope_grid = new vector< vector <float> >;

        for (int row=0; row<nrow; row++){
                for (int col=0; col<ncol; col++){

                }
        }
}
