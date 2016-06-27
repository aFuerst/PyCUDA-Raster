import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from gpustruct import GPUStruct
from multiprocessing import Process,Condition, Lock
import memoryInitializer

def run(input_file, output_file):
  
  ncols, nrows, cellsize, NODATA, xllcorner, yllcorner = getColsRows(input_file)
  
  mem = memoryInitializer.memoryInitializer(ncols, nrows)
  
  load_proc = Process(target=load_func, args=(input_file, mem.to_gpu_buffer, mem.to_gpu_buffer_lock, mem.to_gpu_buffer_full, mem.maxPossRows))
  calc_proc = Process(target=calc_func, args=(output_file, mem, np.int64(nrows), np.int64(ncols), np.float64(cellsize), np.float64(NODATA)))
  
  header_str = ("ncols %s\n"
                "nrows %s\n"
                "xllcorner %s\n"
                "yllcorner %s\n"
                "cellsize %f\n"
                "NODATA_value %d\n"
                % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
               )
  
  write_proc = Process(target=write_func, args=(output_file, header_str, mem.from_gpu_buffer, mem.from_gpu_buffer_lock, mem.from_gpu_buffer_full, nrows))
  
  load_proc.start()
  calc_proc.start()
  write_proc.start()
  
  write_proc.join()


def load_func(input_file, to_gpu_buffer, to_gpu_buffer_lock, to_gpu_buffer_full, maxPossRows):
  f = open(input_file)
  # skip over header
  for i in range(6):
    f.readline()
    
  for line in f:
    
    to_gpu_buffer_lock.acquire()
    while(to_gpu_buffer_full):
      to_gpu_buffer_lock.wait()
      
      
    # fill buffer with data
    for row in range(len(to_gpu_buffer)):
      rw = np.float64(line.split())
      for col in range(len(to_gpu_buffer[0])):
	to_gpu_buffer[row][col] = rw[col]
    
    to_gpu_buffer_full = True
    to_gpu_buffer_lock.notify()
    
    to_gpu_buffer_lock.release()
	
	

def calc_func(output_file, mem, nrows, ncols, cellsize, NODATA):
   
  mod = SourceModule("""
    #include <math.h>
    #include <stdio.h>

    typedef struct{
	    double pixels_per_thread;
	    double NODATA;
	    long ncols;
	    long nrows;
	    long npixels;
    } passed_in;

    /************************************************************************************************
	    GPU only function that gets the neighbors of the pixel at curr_offset
	    stores them in the passed-by-reference array 'store'
    ************************************************************************************************/
    __device__ int getKernel(double *store, double *data, unsigned curr_offset, passed_in *file_info){
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
	    CUDA Kernel function to calculate the slope of pixels in 'data' and stores them in 'result'
	    handles a variable number of calculations based on its thread/block location 
	    and the size of pixels_per_thread in file_info
	    
	    TODO: Create formule to utilize multiple blocks and threads in the y direction
		    see if these changes are able to increase speed
    ************************************************************************************************/
    __global__ void simple_slope(double *data, double *result, passed_in *file_info){
	    /* get individual thread x,y values */
	    unsigned long x = (blockIdx.x * blockDim.x + threadIdx.x)* file_info -> pixels_per_thread;
	    unsigned long y = blockIdx.y * blockDim.y + threadIdx.y; /* always 0 currently */
	    unsigned long offset = x + y;
	    unsigned long i;
	    /* list to store 3x3 kernel each pixel needs to calc slope */
	    double nbhd[9];
	    /* iterate over assigned pixels and calculate slope for all of them */
	    for(i=offset; i < offset + file_info -> pixels_per_thread && i < file_info -> npixels; ++i){
		    if(data[i] == file_info -> NODATA){
			    result[i] = file_info -> NODATA;
		    } else {
			    int q = getKernel(nbhd, data, i, file_info);
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
  
  while(True):
    grid=(4, 4)
    block = (32, 32, 1) 
    num_blocks = grid[0] * grid[1]
    threads_per_block = block[0] * block[1] * block[2]
    pixels_per_thread = np.ceil((nrows * ncols) / (threads_per_block * num_blocks))
    #print "pixels per thread:%d" % pixels_per_thread
    
    # create struct to pass information to C code
    stc = GPUStruct([
      (np.float64, 'pixels_per_thread', pixels_per_thread),
      (np.float64, 'NODATA', NODATA),
      (np.int64, 'ncols', ncols),
      (np.int64, 'nrows', nrows),
      (np.int64, 'npixels', ncols*nrows),
      ])
    stc.copy_to_gpu()
  
    mem.to_gpu_buffer_lock.acquire()
    while(not mem.togpu_buffer_full):
      mem.to_gpu_buffer_lock.wait()
    
    mem.moveToGPU()
    
    # relase lock on buffer
    mem.to_gpu_buffer_full = False
    mem.to_gpu_buffer_lock.notify()
    mem.to_gpu_buffer_lock.release()
    

    
    mem.funcCall(func, stc, block, grid)
    
    mem.from_gpu_buffer_Lock.acquire()
    while(mem.from_gpu_buffer_full):
      mem.from_gpu_buffer_Lock.wait()
      
    mem.getFromGPU()
    
    mem.from_gpu_buffer_full = True
    mem.from_gpu_buffer_Lock.notify()
    mem.from_gpu_buffer_Lock.release()
    
  

def write_func(output_file, header, from_gpu_buffer, from_gpu_buffer_lock, from_gpu_buffer_full, nrows):
  f = open(output_file, 'w')
  f.write(header)
  
  while nrows > 0:
    
    from_gpu_buffer_lock.acquire()
    while not from_gpu_buffer_full:
      from_gpu_buffer_lock.wait()
      
    for row in from_gpu_buffer:
      for col in row:
	f.write(col)
	f.write(' ')
      f.write('\n')
    
    f.flush()
    nrows-=len(from_gpu_buffer)

    from_gpu_buffer_full = False
    from_gpu_buffer_lock.notify()
    from_gpu_buffer_lock.release()
    

def getColsRows(file):
  f = open(file, 'r')
  ncols = np.int64(f.readline().split()[1])
  nrows = np.int64(f.readline().split()[1])
  xllcorner = f.readline().split()[1]
  yllcorner = f.readline().split()[1]
  cellsize = np.float64(f.readline().split()[1])
  NODATA = np.float64(f.readline().split()[1])
  f.close()
  return ncols, nrows, cellsize, NODATA, xllcorner, yllcorner


if __name__ == '__main__':
  run("aigrid.asc", "output.asc")