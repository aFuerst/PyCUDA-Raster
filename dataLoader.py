from osgeo import gdal
from multiprocessing import Process,Condition, Lock
import memoryInitializer
from osgeo import gdal
import struct, os
import numpy as np

class dataLoader(Process):

  def __init__(self, input_file, memInit):
    self.file_name = input_file
    self.open_file = openFile()
    self.header_info = getHeaderInfo()
    self.mem = memInit

  def getHeaderInfo(self):
    return header_info

  def _readHeaderInfo(self):
    if ".asc" in file:
      #f = open(file, 'r')
      ncols = np.int64(f.readline().split()[1])
      nrows = np.int64(f.readline().split()[1])
      xllcorner = f.readline().split()[1]
      yllcorner = f.readline().split()[1]
      cellsize = np.float64(f.readline().split()[1])
      NODATA = np.float64(f.readline().split()[1])
      #f.close()
    else if ".tif" in file:
      #src_ds = gdal.Open("aigrid.tif")
      srcband = src_ds.GetRasterBand(1)
      GeoT = src_ds.GetGeoTransform()
      NODATA = srcband.GetNoDataValue()
      xllcorner = GeoT[0]
      yllcorner = GeoT[3]
      cellsize = srcband.GetScale()
      nrows = srcband.YSize
      ncols = srcband.XSize
    return ncols, nrows, cellsize, NODATA, xllcorner, yllcorner
  
  def _openFile(self):
    if ".asc" in self.file_name:
      f=open(file_name, 'r')
    else if ".tif" in self.file_name:
      f=gdal.Open(self.file_name)
    return f

  def stop(self):
    pass

  def start(self):
    pass

  def loadFunc(self):
    count = self.mem.totalRows
    cur_line = np.zeros(mem.totalCols)
    cur_line.fill(NODATA)
    prev_last_row = np.zeros(mem.totalCols)
    prev_last_row.fill(NODATA)
    while count > 0:
      self.mem.to_gpu_buffer_lock.acquire()

      # Wait until page is emptied
      while mem.to_gpu_buffer_full.is_set():
        self.mem.to_gpu_buffer_lock.wait()

      #Insert last 2 rows of last iteration as first two in this iteration
      for col in range(len(self.mem.to_gpu_buffer[0])):
        self.mem.to_gpu_buffer[0][col] = prev_last_row[col]
      for col in range(len(self.mem.to_gpu_buffer[1])):
        self.mem.to_gpu_buffer[1][col] = cur_line[col]

      # Grab a page worth of input data
      for row in range(self.mem.maxPossRows):
        cur_str = f.readline()

        #Reached end of file
        if cur_str == '':
          #fill rest with NODATA
          while row < mem.maxPossRows:
            for col in range(len(self.mem.to_gpu_buffer[row])):
              self.mem.to_gpu_buffer[row][col] = NODATA
            row += 1
          break

        prev_last_row = cur_line.copy()
        cur_line = np.float64(cur_str.split())
        for col in range(2, len(self.mem.to_gpu_buffer[row])):
          self.mem.to_gpu_buffer[row][col] = cur_line[col]

      # Notify that page is full
      self.mem.to_gpu_buffer_full.set()
      self.mem.to_gpu_buffer_lock.notify()
      self.mem.to_gpu_buffer_lock.release()
      count -= self.mem.maxPossRows

    print "entire file loaded"
