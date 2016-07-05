from multiprocessing import Process,Pipe
import numpy as np
from gpustruct import GPUStruct

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

"""
GPUCalculator

Class that takes and sends data from pipes and goes GPU calculations on it
designed to run as a separate process and inherits from Process module

currently supported functions: slope
"""
class GPUCalculator(Process):
  
    """
    __init__

    paramaters:
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
        _inputPipe - a Pipe object to read information from
        _outputPipe - a Pipe object to send information to

    creates empty instance variables needed later
    """
    def __init__(self, header, _inputPipe, _outputPipe):
        Process.__init__(self)

        # CUDA device info
        self.device = None
        self.context = None

        self.inputPipe = _inputPipe
        self.outputPipe = _outputPipe 

        #unpack header info
        self.totalCols = header[0]
        self.totalRows = header[1]
        self.cellsize = header[2]
        self.NODATA = header[3]

        # memory information
        self.to_gpu_buffer = None
        self.from_gpu_buffer = None
        self.data_gpu = None
        self.result_gpu = None

        #CUDA kernel to be run
        self.kernel = None
        self.func = None

        #carry over rows used to insert last two lines of data from one page
        #as first two lines in next page
        self.carry_over_rows = [np.zeros(self.totalCols), np.zeros(self.totalCols)]
        self.carry_over_rows[0].fill(self.NODATA)
        self.carry_over_rows[1].fill(self.NODATA)

    """
    run

    Overrides default Process.run()
    Given a kernel type, retrieves the C code for that kernel, and runs the
    data processing loop

    does CUDA initialization and sets local device and context
    """
    def run(self, kernelType='simple slope'):
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()

        self._gpuAlloc()

        self.kernel = self.get_kernel(kernelType)
        self.func = self.kernel.get_function("raster_function")

        #Process data while we continue to receive input
        count = 0
        while self.recv_data(count):
            #self.get_kernel(kernelType)
            self.process_data()
            self.write_data(count)
            count += (self.maxPossRows-2)  # -2 because of buffer rows
            print "one iteration done"
        #Process remaining data in buffer
        self.process_data()
        self.write_data(count)

        print "done on GPU"

    """
    _gpuAlloc

    determines how much free memory is on the GPU and allocates as much as needed
    creates pagelocked buffers of equal size to GPU memory
    """
    def _gpuAlloc(self):
        #Get GPU information
        self.freeMem = cuda.mem_get_info()[0] * .5 * .8
        self.maxPossRows = np.int(np.floor(self.freeMem / (8 * self.totalCols)))
        # set max rows to smaller number to save memory usage
        if self.totalRows < self.maxPossRows:
            print "reducing max rows to reduce memory use on GPU"
            self.maxPossRows = self.totalRows

        # create pagelocked buffers and GPU arrays
        self.to_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float64)
        self.from_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float64)
        self.data_gpu = cuda.mem_alloc(self.to_gpu_buffer.nbytes)
        self.result_gpu = cuda.mem_alloc(self.from_gpu_buffer.nbytes)

    """
    recv_data

    Receives a page worth of data from the input pipe. The input pipe comes
    from dataLoader.py. Copies over 2 rows from the previous page so the GPU 
    kernel computation works correctly.
    If the pipe closes, fill the rest of the page with NODATA, and return false
    to indicate that we should break out of the processing loop.
    """
    def recv_data(self, count):
        if count == 0:
            #If this is the first page, insert a buffer row
            for col in range(self.totalCols):
                self.to_gpu_buffer[0][col] = self.carry_over_rows[0][col]
            row_count = 1           
        else:
            #otherwise, insert carry over rows from last page
            for col in range(self.totalCols):
                self.to_gpu_buffer[0][col] = self.carry_over_rows[0][col]
                self.to_gpu_buffer[1][col] = self.carry_over_rows[1][col]
            row_count = 2

        #Receive a page of data from buffer
        while row_count <  self.maxPossRows:
            try:
                # check if something is in the pipe for 5 seconds
                if self.inputPipe.poll(5):
                    cur_row = self.inputPipe.recv()
                else:
                    # end of file reached
                    raise EOFError

                for col in range(self.totalCols):
                    self.to_gpu_buffer[row_count][col] = cur_row[col]

            #Pipe was closed, no more input data
            except EOFError:
                #Fill rest of page with NODATA
                while row_count < self.maxPossRows:
                    for col in range(self.totalCols):
                        self.to_gpu_buffer[row_count][col] = self.NODATA
                    row_count += 1
                return False

            row_count += 1

        #Update carry over rows
        np.put(self.carry_over_rows[0], [i for i in range(self.totalCols)], self.to_gpu_buffer[self.maxPossRows-2])
        np.put(self.carry_over_rows[1], [i for i in range(self.totalCols)], self.to_gpu_buffer[self.maxPossRows-1])

        return True


    """
    process_data

    Using the given kernel code packed in mod, allocates memory on the GPU,
    copies input data from a pagelocked buffer, runs the kernel and copies 
    the output to a second pagelocked buffer
    """
    def process_data(self):
        #GPU layout information
        #func = self.kernel.get_function("raster_function")
        grid = (4,4)
        block = (32,32,1)
        num_blocks = grid[0] * grid[1]
        threads_per_block = block[0]*block[1]*block[2]
        pixels_per_thread = np.ceil((self.maxPossRows * self.totalCols) / (threads_per_block * num_blocks))

        #information struct passed to GPU
        stc = GPUStruct([
            (np.float64, 'pixels_per_thread', pixels_per_thread),
            (np.float64, 'NODATA', self.NODATA),
            (np.uint64, 'ncols', self.totalCols),
            (np.uint64, 'nrows', self.maxPossRows),
            (np.uint64, 'npixels', self.maxPossRows*self.totalCols),
            ])

        stc.copy_to_gpu()

        #Copy input data to GPU
        cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
        #Call GPU kernel
        self.func(self.data_gpu, self.result_gpu, stc.get_ptr(), block=block, grid=grid)
        #Get data back from GPU
        cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu)


    """
    write_data

    Writes results to output pipe. This pipe goes to dataSaver.py
    """
    def write_data(self, count):
        #skip first and last rows, since they were buffers in the computation
        for row in range(1, self.maxPossRows-1):
            # see if written out more than total number of rows plus a small buffer
            # pipe.send seems to sopt working after too many sends and nothing is taken off
            if count + row > self.totalRows:
                print "done", row
                break
            self.outputPipe.send(self.from_gpu_buffer[row])

    """
    stop 

    Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping..."
        exit(1)

    """
    get_kernel

    given a string argument, packages a module for that kernel.
    """
    # NOTE: To create another kernel, add another if statement checking for
    # the string you will identify the kernel by. Then return a SourceModule
    # containing that kernel. Currently, our input/output code in recv_data
    # and write_data assumes that the kernel will treat the first and last
    # row of a given page as buffers that won't be written out.
    # HOWEVER, recv_data is set up so that the last two rows of the preceeding
    # page are used as the first two in the current one. This ensures that the
    # last row of the preceeding page will still be analyzed.
    def get_kernel(self, kernelType):
        if kernelType == 'simple slope':
            mod = SourceModule("""
                    #include <math.h>
                    #include <stdio.h>

                    typedef struct{
                            double pixels_per_thread;
                            double NODATA;
                            unsigned long long ncols;
                            unsigned long long nrows;
                            unsigned long long npixels;
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
                    __global__ void raster_function(double *data, double *result, passed_in *file_info){
                            /* get individual thread x,y values */
                            unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
                            unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y; 
                            unsigned long long offset = (gridDim.x*blockDim.x) * y + x; 
                            //gridDim.x * blockDim.x is the width of the grid in threads. This moves us to the correct
                            //block and thread.
                            unsigned long long i;
                            /* list to store 3x3 kernel each pixel needs to calc slope */
                            double nbhd[9];
                            /* iterate over assigned pixels and calculate slope for all of them */
                            /* do npixels + 1 to make last row(s) get done */
                            for(i=0; i < file_info -> pixels_per_thread + 1 && offset < file_info -> npixels; ++i){	    
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
            return mod
        else:
            print "CUDA kernel not implemented"
            self.stop()
