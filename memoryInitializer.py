import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from gpustruct import GPUStruct
import numpy as np

"""
provides automatic allocation of available memory based on file size to be computed

"""
class memoryInitializer:
  
  """
  takes 2 parameters, the number of rows and columns of the file to be computed
  allocates aximum available space on GPU and host for input and output buffers on each
  allocated memory will be a multiple of the maximum number of rows that can be computed at once
  """
  def __init__(self, totalCols, totalRows):
    self.totalCols = totalCols
    self.totalRows = totalRows
    
    # get information about memory currently available on GPU
    self.freeMem = cuda.mem_get_info()[0]
    self.totalMem = cuda.mem_get_info()[1]
    
    # determine how many full rows can fit on device with available memory
    gpu_buffer_size = freeMem * .5 * .8
    self.maxPossRows = np.floor(gpu_buffer_size / totalCols)
    
    # allocate host memory to use to transfer data to GPU
    self.to_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , totalCols), np.float64)
    self.from_gpu_buffer = cuda.pagelocked_zeros((self.maxPossRows , totalCols), np.float64)
  
    # allocate device memory 
    self.data_gpu = cuda.mem_alloc(to_gpu_buffer.nbytes)
    self.result_gpu = cuda.mem_alloc(from_gpu_buffer.nbytes)
    
    # send empty result array to GPU
    cuda.memcpy_htod(result_gpu, from_gpu_buffer)

    
  """
  frees all allocated memory on host and GPU
  """
  def __del__(self):
    self.to_gpu_buffer.free().
    self.from_gpu_buffer.free()
    self.data_gpu.free()
    self.result_gpu.free()
  
  # returns the maximun number of full rows that can fit on the GPU
  def getMaxPossRows(self):
    return self.maxPossRows
  
  #returns amount of free memory available on GPU when object was created
  def getFreeMem(self):
    return self.freeMem
  
  # copies the data in to_gpu_buffer to the GPU
  def moveToGPU(self):
    cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
  
  # cpoies the data from the GPU to from_gpu_buffer and returns it
  def getfromGPU(self):
    cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu)
    return self.from_gpu_buffer
  
  # returns the 2d array of pagelocked memory used to pass data to the GPU
  # gives ability to manually set values in array
  def getGPUBuffer(self):
    return to_gpu_buffer
  
  # executes function call 'func' on data_gpu and result_gpu and other parameters passed in
  def funcCall(self, func, struct, block, grid):
      func(data_gpu, result_gpu, struct.get_ptr(), block=block, grid=grid)