from osgeo import gdal
from multiprocessing import Process,Condition, Lock
import memoryInitializer
from osgeo import gdal
import struct, os
import numpy as np

"""
currently supported file types: [GEOTiff (.tif), ESRI ASCII format (.asc)]
"""
class dataLoader(Process):

    """
    """
    def __init__(self, input_file):
        Process.__init__(self)
        self.file_name = input_file
        self.open_file = _openFile()
        self.header_info = _readHeaderInfo()
        self.mem = None
        self.cur_line=""
        self.prev_last_row=""
        self.stopBool = False

    """
        Returns header information as a six-tuple in this order:
        (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    """
    def getHeaderInfo(self):
        return header_info

    """
    """
    def setMemInit(self, mem):
        self.mem = mem

    """
        uses open file object and returns this header information:
        (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    """
    def _readHeaderInfo(self):
        if ".asc" in file:
            ncols = np.int64(self.open_file.readline().split()[1])
            nrows = np.int64(self.open_file.readline().split()[1])
            xllcorner = self.open_file.readline().split()[1]
            yllcorner = self.open_file.readline().split()[1]
            cellsize = np.float64(self.open_file.readline().split()[1])
            NODATA = np.float64(self.open_file.readline().split()[1])
        else if ".tif" in file:
            srcband = src_ds.GetRasterBand(1)
            GeoT = src_ds.GetGeoTransform()
            NODATA = srcband.GetNoDataValue()
            xllcorner = GeoT[0]
            yllcorner = GeoT[3]
            cellsize = srcband.GetScale()
            nrows = srcband.YSize
            ncols = srcband.XSize
        return ncols, nrows, cellsize, NODATA, xllcorner, yllcorner
  
    """
        opens file_name and sets it to open_file
    """
    def _openFile(self):
        if ".asc" in self.file_name:
            self.open_file=open(file_name, 'r')
        else if ".tif" in self.file_name:
            self.open_file=gdal.Open(self.file_name)

    """
        Alerts the thread that it needs to quit
    """
    def stop(self):
      self.stopBool=True

    """
    """
    #def run(self):

    """
    """
    def _getLine(self, row):
        if ".asc" in self.file_name:
            f=self.open_file.readline()
        else if ".tif" in self.file_name:
            try:
                ncols = self.header_info[0]
                fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}
                f=struct.unpack(fmttypes[gdal.GetDataTypeName(self.open_file.DataType)]*ncols, srcband.ReadRaster(0,row,srcband.XSize,1, buf_type=self.open_file.DataType))
            except gdal.RuntimeError:
                f=[]   
        return np.float64(f)
        

    """
    """
    def _loadFunc(self):
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
                cur_str = self._getline(count+row)

                #Reached end of file
                if cur_str == []:
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


