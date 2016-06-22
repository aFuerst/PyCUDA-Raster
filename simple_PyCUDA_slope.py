import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from gpustruct import GPUStruct
import numpy as np

def run():

    input_file = open("aigrid.asc", 'r')
    ncols = np.float64(input_file.readline().split()[1])
    nrows = np.float64(input_file.readline().split()[1])
    xllcorner = input_file.readline().split()[1]
    yllcorner = input_file.readline().split()[1]
    cellsize = np.float64(input_file.readline().split()[1])
    NODATA = np.float64(input_file.readline().split()[1])

    numberBlocks = 1
    print "number of blocks:%d" % numberBlocks

    total_pixels = np.float64(ncols * nrows)
    pixels_per_thread = np.float64(total_pixels / (1024 * numberBlocks))
    print "pixels per thread:%d" % pixels_per_thread

    # read data in as n by m list of np floats
    # NOTE: Don't skip any lines here, the file pointer has already advanced
    # past the header to the data.
	    
    data = np.loadtxt(input_file)
    input_file.close()
    data = data.astype(np.float64)
    result = np.empty_like(data)

    # allocate needed data space into GPU
    data_gpu = cuda.mem_alloc(data.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    # transfer data to GPU
    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(result_gpu, result)

    stc = GPUStruct([
	    (np.float64, 'pixels_per_thread', pixels_per_thread),
	    (np.float64, 'NODATA', NODATA),
	    (np.int64, 'ncols', ncols),
	    (np.int64, 'nrows', nrows),
	    (np.int64, 'npixels', ncols*nrows),
	    (np.int64, 'count', 0)
    ])
    stc.copy_to_gpu()

    # function(s) to be compiled by CUDA for execution on GPU
    mod = SourceModule(
    """
    #include <math.h>
    #include <stdio.h>

    typedef struct{
	    double pixels_per_thread;
	    double NODATA;
	    long ncols;
	    long nrows;
	    long npixels;
	    long count;
    } passed_in;

    /************************************************************************************************
	    on-GPU function that gets the neighbors of the pixel at curr_offset
	    stores them in the passed-by-reference array 'store'
    ************************************************************************************************/
    __device__ int getNeighbors(double *store, double *data, unsigned curr_offset, passed_in *file_info){
	    int i;
	    for(i = -1; i < 2; i++){
		    if((i + curr_offset - file_info -> ncols) < 0){
		      store[i + 1] = data[curr_offset];
		    } else {
		      store[i + 1] = data[i + curr_offset - (int)(file_info -> ncols)];
		    }
		    
		    store[i + 4] = data[i + curr_offset];
		    
		    if((i + curr_offset + file_info -> ncols) > file_info -> npixels){
		      store[i + 7] = data[i + curr_offset + (int)(file_info -> ncols)];
		    } else {
		      store[i + 7] = data[curr_offset];
		    }
	    }
	    /* return a value otherwise it throws a warning expression not having effect */
	    return i;
    }

    /************************************************************************************************
	    kernel function to calculate the slope of pixels in 'data' and stores them in 'result'
	    handles a variable number of calculations based on its thread/block location 
	    and the size of pixels_per_thread
	    
	    TODO: Create formule to utilize multiple blocks and threads in the y direction
		    see if these changes are able to increase speed
    ************************************************************************************************/
    __global__ void simple_slope(double *data, double *result, passed_in *file_info){
	    /* get individual thread x,y coordinates */
	    /* probably not correct */
	    unsigned long x = blockIdx.x * blockDim.x + threadIdx.x* file_info -> pixels_per_thread;
	    unsigned long y = blockIdx.y * blockDim.y + threadIdx.y;
	    unsigned long offset = x + y;
	    unsigned long i;
	    /* iterate over assigned pixels and calculate slope for all of them */
	    double nbhd[9];
	    for(i=offset; i < offset + file_info -> pixels_per_thread && i < file_info -> npixels; ++i){
		    if(data[i] == file_info -> NODATA){
			    result[i] = file_info -> NODATA;
		    } else {
			    int q = getNeighbors(nbhd, data, i, file_info);
			    for(q = 0; q < 9; ++q){
				    if(nbhd[q] == file_info -> NODATA){
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

    func = mod.get_function("simple_slope")

    # call slope function using a 2x3 grid of blocks which are made up of 16x16x1 threads -> 
    #func(data_gpu, result_gpu, stc.get_ptr(), block = (1, 1, 1), grid=(1, 1))

    func(data_gpu, result_gpu, stc.get_ptr(), block = (1024, 1, 1), grid=(1,1))

    # copy result back to host
    cuda.memcpy_dtoh(result, result_gpu)
    stc.copy_from_gpu()
    print stc

    # free on-device memory, not necessary currently; just here to test if it works
    # not sure if we will need to do this when dealing with files bigger than space on GPU
    #result_gpu.free()
    #data_gpu.free()

    # write result to file
    header_str = ("ncols %s\n"
		  "nrows %s\n"
		  "xllcorner %s\n"
		  "yllcorner %s\n"
		  "cellsize %f\n"
		  "NODATA_value %d"
		  % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
		)

    np.savetxt("output.asc", result, fmt='%5.2f', header=header_str, comments='')


if __name__ == '__main__':
    import timeit
    print (timeit.timeit("run()",setup="from __main__ import run", number=1))
    
    
    
    