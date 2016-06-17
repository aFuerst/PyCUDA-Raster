import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

input_file = open("aigrid.asc", 'r')
ncols = numpy.float(input_file.readline().split()[1])
nrows = numpy.float(input_file.readline().split()[1])
xllcorner = input_file.readline().split()[1]
yllcorner = input_file.readline().split()[1]
cellsize = numpy.float(input_file.readline().split()[1])
NODATA = numpy.float(input_file.readline().split()[1])

total_pixels = numpy.float(ncols * nrows)
pixels_per_thread = numpy.float(total_pixels / (16 * 16 * 5))

    # read data in as n by m list of numpy floats
    # NOTE: Don't skip any lines here, the file pointer has already advanced
    # past the header to the data.
	
data = np.loadtxt(input_file)
input_file.close()
result = numpy.empty_like(data)
result = result.astype(numpy.float64)
data = data.astype(numpy.float64)

# allocate needed data space into GPU
data_gpu = cuda.mem_alloc(data.nbytes)
result_gpu = cuda.mem_alloc(result.nbytes)
ppt_gpu = cuda.mem_alloc(pixels_per_thread.nbytes)
NODATA_gpu = cuda.mem_alloc(NODATA.nbytes)
ncols_gpu = cuda.mem_alloc(ncols.nbytes)
nrows_gpu = cuda.mem_alloc(nrows.nbytes)

# transfer data to GPU
htod(data_gpu, data)
htod(result_gpu, result)
htod(ppt_gpu, pixels_per_thread)
htod(NODATA_gpu, NODATA)
htod(ncols_gpu, ncols)
htod(nrows_gpu, nrows)

# function(s) to be compiled by CUDA for execution on GPU
mod = SourceModule(
"""
#include <math.h>

/************************************************************************************************
	on-GPU function that gets the neighbors of the pixel at curr_offset
	stores them in the passed-by-reference array 'store'
 ************************************************************************************************/
__device__ getNeighbors(double *store, double *data, unsigned curr_offset, int pixels_per_thread){
	int i;
	for(i = -1, i < 2, ++i){
		store[1 + i] = data[curr_offset + i];
		store[4 + i] = data[curr_offset + i + 4];
		store[7 + i] = data[curr_offset + i + 7];
	}
}

/************************************************************************************************
	kernel function to calculate the slope of pixels in 'data' and stores them in 'result'
	handles a variable number of calculations based on its thread/block location and the size of pixels_per_thread
	
	TODO: find a way to use a struct to hold (pixels_per_thread, NODATA, ncols, nrows) to cut down on parameters
		  Ensure function is actually calculating all pixels and isn;t missing any,
			unsure if offset calculation is correct

************************************************************************************************/
__global__ void simple_slope(double *data, double *result, int pixels_per_thread, double NODATA, int ncols, int nrows){
	/* get individual thread x,y coordinates */
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = x * y * pixels_per_thread;

	int i;
	/* iterate over assigned pixels and calculate slope for all of them */
	for(i=offset; i < offset + pixels_per_thread, ++i){

		if(data[i] == NODATA){
			result[i] == NODATA;
		} else{
			double nbhd[9];
			getNeighbors(nbhd, data, i, pixels_per_thread);
			int q = 0;
			for(q, q < 9, ++q){
				if(nbhd[q] == NODATA){
					nbhd[q] = data[i];
				}
			}
			double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*10);
			double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*10);
			result[i] = atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));	
		}
	}
}
""")

func = mod.get_function("slope")

# call slope function using 5 16x16 blocks pf threads -> 
func(data_gpu, result_gpu, ppt_gpu, NODATA_gpu, ncols_gpu, nrows_gpu, block = (16, 16, 5))

# copy result back to host
cuda.memcpy_dtoh(result, result_gpu)

# free on-device memory, not necessary currently; just here to test if it works
# not sure if we will need to do this when dealing with files bigger than space on GPU
result_gpu.free()
data_gpu.free()
ppt_gpu.free()
NODATA_gpu.free()

# write result to file
header_str = ("ncols %s\n"
              "xllcorner %s\n"
              "nrows %s\n"
              "yllcorner %s\n"
              "cellsize %f\n"
              "NODATA_value %d"
              % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
             )
np.savetxt("output.asc", result, fmt='%5.2f', header=header_str, comments='')


