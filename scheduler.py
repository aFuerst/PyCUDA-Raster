#import pycuda.autoinit
#import pycuda.driver as cuda
#from pycuda.compiler import SourceModule
import numpy as np
from gpustruct import GPUStruct
from multiprocessing import Process,Condition, Lock
import memoryInitializer

def run(input_file, output_file):
  
  ncols, nrows, cellsize, NODATA, xllcorner, yllcorner = getColsRows(input_file)
  
  mem = memoryInitializer.memoryInitializer(ncols, nrows)

  load_proc = Process(target=load_func, args=(input_file, mem))
  #calc_proc = Process(target=calc_func, args=(mem, np.int64(nrows), np.int64(ncols), np.float64(cellsize), np.float64(NODATA)))
  
  header_str = ("ncols %s\n"
                "nrows %s\n"
                "xllcorner %s\n"
                "yllcorner %s\n"
                "cellsize %f\n"
                "NODATA_value %d\n"
                % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
               )
  
  write_proc = Process(target=write_func, args=(output_file, header_str, mem, nrows))
  
  load_proc.start()
  #calc_proc.start()
  write_proc.start()
  
  calc_func(mem, np.int64(nrows), np.int64(ncols), np.float64(cellsize), np.float64(NODATA))
  
  load_proc.join()  
  write_proc.join()
  
  print "Processing completed"
  
def load_func(input_file, mem):
  f = open(input_file)
  # skip over header
  for i in range(6):
    f.readline()
 
  cur_line = ' ' #current input line
  while cur_line != '':
    mem.to_gpu_buffer_lock.acquire()
    #Wait until page is emptied
    while mem.to_gpu_buffer_full.is_set():
      #print "waiting", mem.to_gpu_buffer_full
      mem.to_gpu_buffer_lock.wait()
      
    #Grab a page worth of input data
    for row in range(mem.maxPossRows):
      cur_line = f.readline()
      if cur_line == '':
	break
      cur_line = np.float64(cur_line.split())
      for col in range(len(mem.to_gpu_buffer[row])):
	mem.to_gpu_buffer[row][col] = cur_line[col]

    #Notify that page is full
    mem.to_gpu_buffer_full.set()
    mem.to_gpu_buffer_lock.notify()
    
    mem.to_gpu_buffer_lock.release()
     
  print "entire file loaded"

def calc_func(mem, nrows, ncols, cellsize, NODATA):
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
  
  #cuda.init()
  #ctx = cuda.Device(0).make_context()
  #device = self.ctx.get_device()
  
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
  decrement = mem.totalRows
  while(decrement > 0):
    print decrement
    decrement-=nrows
    
    print "here"
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
    while(not mem.to_gpu_buffer_full.is_set()):
      #print "waitin 2", mem.to_gpu_buffer_full
      mem.to_gpu_buffer_lock.wait()
    
    mem.moveToGPU()
    
    # relase lock on buffer
    mem.to_gpu_buffer_full.clear()
    mem.to_gpu_buffer_lock.notify_all()
    mem.to_gpu_buffer_lock.release()
    
    print "done calculations"
    
    mem.funcCall(func, stc, block, grid)
    
    mem.from_gpu_buffer_lock.acquire()
    while(mem.from_gpu_buffer_full.is_set()):
      mem.from_gpu_buffer_lock.wait()
      
     
    mem.getFromGPU()
    
    print "data returned from GPU"
    
    mem.from_gpu_buffer_full.set()
    mem.from_gpu_buffer_lock.notify()
    mem.from_gpu_buffer_lock.release()
    
  print "done computing on gpu"

def write_func(output_file, header, mem, nrows):
  f = open(output_file, 'w')
  f.write(header)
  
  while nrows > 0:
    
    mem.from_gpu_buffer_lock.acquire()
    while not mem.from_gpu_buffer_full.is_set():
      mem.from_gpu_buffer_lock.wait()
      
    for row in mem.from_gpu_buffer:
      for col in row:
	f.write(str(col))
	f.write(' ')
      f.write('\n')
    
    f.flush()
    nrows-=len(mem.from_gpu_buffer)

    mem.from_gpu_buffer_full.clear()
    mem.from_gpu_buffer_lock.notify()
    mem.from_gpu_buffer_lock.release()
    
  print "done writing to file"

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