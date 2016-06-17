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
data_gpu = htod(data_gpu, data)
result_gpu = htod(result_gpu, result)
ppt_gpu = htod(ppt_gpu, pixels_per_thread)
NODATA_gpu = htod(NODATA_gpu, NODATA)
ncols_gpu = htod(ncols_gpu, ncols)
nrows_gpu = htod(nrows_gpu, nrows)

mod = SourceModule(
"""
#include <math.h>

__global__ void simple_slope(double *data, double *result, int pixels_per_thread, double NODATA, int ncols, int nrows){
	/* get thread x,y coordinates */
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = x * y * pixels_per_thread;

	int i;
	for(i=offset; i < offset + pixels_per_thread, ++i){
		/*
		if(data[i] == NODATA){
			result[i] == NODATA;
		} else{
			double nbhd[9];
			int q = 0;
			for(q, q < 9, ++q){
				if(data[i+q] == NODATA){
					nbhd[q] = data[i+4];
				} else {
					nbhd[q] = data[i+q]; 
				}
			}
			double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*10);
			double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*10);
			result[i] = atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
			
		}
		*/
		result[i] = offset;
	}
}
""")

func = mod.get_function("slope")

# call slope function using 5 16x16 blocks pf threads -> 
func(data_gpu, result_gpu, ppt_gpu, NODATA_gpu, ncols_gpu, nrows_gpu, block = (16, 16, 5))

cuda.memcpy_dtoh(result, result_gpu)

# free on-device memory
result_gpu.free()
data_gpu.free()
ppt_gpu.free()
NODATA_gpu.free()

header_str = ("ncols %s\n"
              "xllcorner %s\n"
              "nrows %s\n"
              "yllcorner %s\n"
              "cellsize %f\n"
              "NODATA_value %d"
              % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
             )

#print slope_data
np.savetxt("output.asc", result, fmt='%5.2f', header=header_str, comments='')