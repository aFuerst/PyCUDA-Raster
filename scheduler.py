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
  
  header_str = ("ncols %s\n"
                "nrows %s\n"
                "xllcorner %s\n"
                "yllcorner %s\n"
                "cellsize %f\n"
                "NODATA_value %d\n"
                % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
               )
  
  # create two processes to load and write data
  load_proc = Process(target=load_func, args=(input_file, mem))
  write_proc = Process(target=write_func, args=(output_file, header_str, mem, nrows))
  
  # start all processes
  load_proc.start()
  write_proc.start()
  
  # run function to calculate in main to maintain context
  calc_func(mem, np.int64(nrows), np.int64(ncols), np.float64(cellsize), np.float64(NODATA))
  
  load_proc.join()  
  write_proc.join()
  
  print "Processing completed"
  
def load_func(input_file, mem):
  f = open(input_file)
  # skip over header
  for i in range(6):
    f.readline()
 
  cur_line = ' ' # current input line
  while cur_line != '':
    mem.to_gpu_buffer_lock.acquire()
    # Wait until page is emptied
    while mem.to_gpu_buffer_full.is_set():
      mem.to_gpu_buffer_lock.wait()
      
    # Grab a page worth of input data
    for row in range(mem.maxPossRows):
      cur_line = f.readline()
      if cur_line == '':
	break
      cur_line = np.float64(cur_line.split())
      for col in range(len(mem.to_gpu_buffer[row])):
	mem.to_gpu_buffer[row][col] = cur_line[col]

    # Notify that page is full
    mem.to_gpu_buffer_full.set()
    mem.to_gpu_buffer_lock.notify()
    
    mem.to_gpu_buffer_lock.release()
     
  print "entire file loaded"

def calc_func(mem, nrows, ncols, cellsize, NODATA):
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule

  # C code to be executed on GPU
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
  __device__ int getKernel(double *store, double *data, unsigned long offset, passed_in *file_info){
          //NOTE: This is more or less appropriated from Liam's code. Treats edge rows and columns
          // as buffers, they will be dropped.
          if (offset < file_info->ncols || offset >= (file_info->npixels - file_info->ncols)){
                return 1;
          }
          unsigned long y = offset % file_info->ncols; //FIXME: I'm not sure why this works...
          if (y == (file_info->ncols - 1) || y == 0){
                return 1;
          }
          // Grab neighbors above and below.
          store[1] = data[offset - file_info->ncols];
          store[7] = data[offset + file_info->ncols];
          // Grab right side neighbors.
          store[2] = data[offset - file_info->ncols + 1];
          store[5] = data[offset + 1];
          store[8] = data[offset + file_info->ncols + 1];
          // Grab left side neighbors.
          store[0] = data[offset - file_info->ncols - 1];
          store[3] = data[offset - 1];
          store[6] = data[offset + file_info->ncols - 1];
          /* return a value otherwise it throws a warning expression not having effect */
          return 0;
  }

  /************************************************************************************************
          CUDA Kernel function to calculate the slope of pixels in 'data' and stores them in 'result'
          handles a variable number of calculations based on its thread/block location 
          and the size of pixels_per_thread in file_info
  ************************************************************************************************/
  __global__ void simple_slope(double *data, double *result, passed_in *file_info){
          /* get individual thread x,y values */
          unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
          unsigned long y = blockIdx.y * blockDim.y + threadIdx.y; 
          unsigned long offset = (gridDim.x*blockDim.x) * y + x; 
          //gridDim.x * blockDim.x is the width of the grid in threads. This moves us to the correct
          //block and thread.
          unsigned long i;
          /* list to store 3x3 kernel each pixel needs to calc slope */
          double nbhd[9];
          /* iterate over assigned pixels and calculate slope for all of them */
          for(i=0; i < file_info -> pixels_per_thread + 1; ++i){
                  if(offset > file_info -> npixels){
                    break;
                  }	    
                  if(data[offset] == file_info -> NODATA){
                          result[offset] = file_info -> NODATA;
                  } else {
                          int q = getKernel(nbhd, data, offset, file_info);
                          if (q) {
                                result[offset] = file_info->NODATA;
                          }
                          else{
                                for(q = 0; q < 9; ++q){
                                        if(nbhd[q] == file_info -> NODATA){
                                                nbhd[q] = data[offset];
                                        }
                                }
                                double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8*10);
                                double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8*10);
                                result[offset] = atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
                          }
                  }
                  offset += (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y);
                  //Jump to next row

          }
  }
  """)
  
  func = mod.get_function("simple_slope")
  
  # counter to keep tack of how many time to run calculation
  decrement = mem.totalRows
  while(decrement > 0):
    decrement-=nrows
    
    # generate grib and block layout for threads
    grid=(4, 4)
    block = (32, 32, 1) 
    num_blocks = grid[0] * grid[1]
    threads_per_block = block[0] * block[1] * block[2]
    pixels_per_thread = np.ceil((nrows * ncols) / (threads_per_block * num_blocks))
    
    # create struct to pass information to C code
    stc = GPUStruct([
      (np.float64, 'pixels_per_thread', pixels_per_thread),
      (np.float64, 'NODATA', NODATA),
      (np.int64, 'ncols', ncols),
      (np.int64, 'nrows', nrows),
      (np.int64, 'npixels', ncols*nrows),
      ])
    stc.copy_to_gpu()

    # wait until buffer is full
    mem.to_gpu_buffer_lock.acquire()
    while(not mem.to_gpu_buffer_full.is_set()):
      mem.to_gpu_buffer_lock.wait()
    
    mem.moveToGPU()
    
    # relase lock on buffer
    mem.to_gpu_buffer_full.clear()
    mem.to_gpu_buffer_lock.notify_all()
    mem.to_gpu_buffer_lock.release()
    
    # make call to CUDA to run GPU 
    mem.funcCall(func, stc, block, grid)
    
    # wait until buffer is empty
    mem.from_gpu_buffer_lock.acquire()
    while(mem.from_gpu_buffer_full.is_set()):
      mem.from_gpu_buffer_lock.wait()
      
    # get data off GPU and put into RAM buffer
    mem.getFromGPU()
    
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
