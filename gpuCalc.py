from multiprocessing import Process,Pipe
import numpy as np
from gpustruct import GPUStruct

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

TOTALROWCOUNT = 0

"""
GPUCalculator

Class that takes and sends data from pipes and goes GPU calculations on it
designed to run as a separate process and inherits from Process module

currently supported functions: slope, aspect, hillshade
"""
class GPUCalculator(Process):
  
    """
    __init__

    paramaters:
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
        _input_pipe - a Pipe object to read information from
        _outputPipe - a Pipe object to send information to
        function_types - list of strings that are supported function names as strings

    creates empty instance variables needed later
    """
    def __init__(self, header, _input_pipe, _output_pipes, function_types):
        Process.__init__(self)

        # CUDA device info
        self.device = None
        self.context = None

        self.input_pipe = _input_pipe
        self.output_pipes = _output_pipes 
        self.functions = function_types
    
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
    def run(self):
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()

        self._gpuAlloc()

        self.kernel = self._getKernel()
        self.func = self.kernel.get_function("raster_function")

        #Process data while we continue to receive input
        count = 0
        while self._recvData(count):
            #Copy input data to GPU
            cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
            for i in range(len(self.functions)):
                self._processData(self.functions[i])
                #Get data back from GPU
                cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu)
                self._writeData(count, self.output_pipes[i])

            count += (self.maxPossRows-2)  # -2 because of buffer rows
            print "Page done... %.3f %% completed" % ((float(count) / float(self.totalRows)) * 100)
        #Process remaining data in buffer
        cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
        for i in range(len(self.functions)):
            self._processData(self.functions[i])
            cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu) 
            self._writeData(count, self.output_pipes[i])

        for pipe in self.output_pipes:
            pipe.close()

        print "GPU calculations finished"

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
            #self.maxPossRows = 100

        # create pagelocked buffers and GPU arrays
        self.to_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float64)
        self.from_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float64)
        self.data_gpu = cuda.mem_alloc(self.to_gpu_buffer.nbytes)
        self.result_gpu = cuda.mem_alloc(self.from_gpu_buffer.nbytes)

    """
    _recvData

    Receives a page worth of data from the input pipe. The input pipe comes
    from dataLoader.py. Copies over 2 rows from the previous page so the GPU 
    kernel computation works correctly.
    If the pipe closes, fill the rest of the page with NODATA, and return false
    to indicate that we should break out of the processing loop.
    """
    def _recvData(self, count):
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
                if count + row_count > self.totalRows:
                    # end of file reached       
                    cur_row = None             
                    for col in range(self.totalCols):
                        self.to_gpu_buffer[row_count][col] = self.NODATA
                    return False
                else:
                    cur_row = self.input_pipe.recv()

                for col in range(self.totalCols):
                    self.to_gpu_buffer[row_count][col] = cur_row[col]

            #Pipe was closed unexpectedly
            except EOFError:
                print "Pipe closed unexpectedly."
                self.stop()

            row_count += 1
            
        #Update carry over rows
        np.put(self.carry_over_rows[0], [i for i in range(self.totalCols)], self.to_gpu_buffer[self.maxPossRows-2])
        np.put(self.carry_over_rows[1], [i for i in range(self.totalCols)], self.to_gpu_buffer[self.maxPossRows-1])

        return True


    """
    _processData

    Using the given kernel code packed in mod, allocates memory on the GPU,
    and runs the kernel.
    """
    def _processData(self, funcType):
        #GPU layout information
        grid = (16,16)
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
            (np.float64, 'cellSize', self.cellsize),
            (np.int32, 'function', self._getFunctionVal(funcType))
            ])

        stc.copy_to_gpu()

        #Call GPU kernel
        self.func(self.data_gpu, self.result_gpu, stc.get_ptr(), block=block, grid=grid)

    """
    _getFunctionVal

    returns the case needed for a particular function to run the proper CUDA
    code
    """
    def _getFunctionVal(self, func):
        if func == "slope":
            return 0        
        elif func == "aspect":
            return 1
        elif func == "hillshade":
            return 2
        else:
            print "Illegal function chosen"
            raise NotImplemented

    """
    _writeData

    Writes results to output pipe. This pipe goes to dataSaver.py
    """
    def _writeData(self, count, out_pipe):
        for row in range(1, self.maxPossRows-1):
            if count + row > self.totalRows:
                break
            out_pipe.send(self.from_gpu_buffer[row])
    """
    stop 

    Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping gpuCalc..."
        self.data_gpu.free()
        self.result_gpu.free()
        exit(1)

    """
    _getKernel

    Packages the kernel module.
    """
    # NOTE: To create another raster function, add another if statement to
    # _getFunctionVal checking for the string you will identify the kernel by,
    # another case to raster_function in getKernal, and the function code.
    # The GPUCalculator class is set up to automatically insert buffer rows at
    # the beginning and end of the file so that all rows are calculated correctly.
    def _getKernel(self):
        mod = SourceModule("""
                    #include <math.h>
                    #include <stdio.h>

                    typedef struct{
                            double pixels_per_thread;
                            double NODATA;
                            unsigned long long ncols;
                            unsigned long long nrows;
                            unsigned long long npixels;
                            double cellSize;
                            int function;
                    } passed_in;

                    /************************************************************************************************
                            GPU only function that gets the neighbors of the pixel at offset
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

                    /*
                        GPU only function that calculates slope for a pixel
                    */
                    __device__ double slope(double dz_dx, double dz_dy){
                        return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
                    }

                    /*
                        GPU only function that calculates aspect for a pixel
                    */
                    __device__ double aspect(double dz_dx, double dz_dy, double NODATA){
                        double aspect = 57.29578 * (atan2(dz_dy, -(dz_dx)));
                        if(dz_dx == NODATA || dz_dy == NODATA || (dz_dx == 0.0 && dz_dy == 0.0)){
                            return NODATA;
                        } else{
                            if(aspect > 90.0){
                                aspect = 360.0 - aspect + 90.0;
                            } else {
                                aspect = 90.0 - aspect;
                            }
                                aspect = aspect * (M_PI / 180.0);
                                return aspect;
                            }
                        }

                    /*
                        GPU only function that calculates aspect for a pixel
                        to be ONLY used by hillshade
                    */    
                    __device__ double hillshade_aspect(double dz_dx, double dz_dy){
                        double aspect;
                        	if(dz_dx != 0){
                        	    aspect = atan2(dz_dy, -(dz_dx));
                        	    if(aspect < 0){
                        	        aspect = ((2 * M_PI) + aspect);
                                }
                            } else if(dz_dx == 0){
                            	if(dz_dy > 0){
                        	        aspect = (M_PI / 2);
                        	    }else if(dz_dy < 0){
                        	        aspect = ((2 * M_PI) - (M_PI / 2));
                        	    }else{
                        	        aspect = atan2(dz_dy, -(dz_dx));
                                }
                            }
                        return aspect;
                    }

                    /*
                        GPU only function that calculates hillshade for a pixel
                    */
                    __device__ double hillshade(double dz_dx, double dz_dy){
                        /* calc slope and aspect */
                        double slp = slope(dz_dx, dz_dy);
                        double asp = hillshade_aspect(dz_dx, dz_dy);

                        /* calc zenith */
	                    double altitude = 45;
	                    double zenith_deg = 90 - altitude;
	                    double zenith_rad = zenith_deg * (M_PI / 180.0);
	
                        /* calc azimuth */
	                    double azimuth = 315;
	                    double azimuth_math = (360 - azimuth + 90);
	                    if(azimuth_math >= 360.0){
		                    azimuth_math = azimuth_math - 360;
                        }	
                        double azimuth_rad = (azimuth_math * M_PI / 180.0);

                        double hs = 255.0 * ( ( cos(zenith_rad) * cos(slp) ) + ( sin(zenith_rad) * sin(slp) * cos(azimuth_rad - asp) ) );

	                    if(hs < 0){
		                    return 0;
                        } else {
                            return hs;
                        }
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
                                        } else {
                                            for(q = 0; q < 9; ++q){
                                                if(nbhd[q] == file_info -> NODATA){
                                                    nbhd[q] = data[offset];
                                                }
                                            }
                                            double dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8 * file_info -> cellSize);
                                            double dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8 * file_info -> cellSize);
                                            /* choose which function to execute */
                                            switch(file_info -> function){
                                                case 0:
                                                    result[offset] = slope(dz_dx, dz_dy);
                                                break;
                                                case 1:
                                                    result[offset] = aspect(dz_dx, dz_dy, file_info -> NODATA);
                                                break;
                                                case 2:
                                                    result[offset] = hillshade(dz_dx, dz_dy);
                                                break;                        
                                            }
                                        }
                                    }
                                    offset += (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y);
                                    //Jump to next row
                            }
                    }
                    """)
        return mod

