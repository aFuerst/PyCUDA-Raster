from osgeo import gdal
from multiprocessing import Process,Condition, Lock, Pipe, Connection
#import memoryInitializer
from osgeo import gdal
import struct, os
import numpy as np

"""
currently supported file types: [GEOTiff (.tif), ESRI ASCII format (.asc)]
"""
class dataLoader(Process):

    """
        
    """
    def __init__(self, input_file, output_pipe):
        Process.__init__(self)
        self.output_pipe = output_pipe
        self.file_name = input_file
        self.open_file = _openFile()
        
        # get header info
        _readHeaderInfo()

        self.cur_line=""
        self.prev_last_row=""
        self.stopBool = False

    """
        Returns header information as a six-tuple in this order:
        (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    """
    def getHeaderInfo(self):
        return self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner

    """
        uses open file object and returns this header information:
        (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    """
    def _readHeaderInfo(self):
        if ".asc" in file:
            self.totalCols = np.int64(self.open_file.readline().split()[1])
            self.totalRows = np.int64(self.open_file.readline().split()[1])
            self.xllcorner = self.open_file.readline().split()[1]
            self.yllcorner = self.open_file.readline().split()[1]
            self.cellsize = np.float64(self.open_file.readline().split()[1])
            self.NODATA = np.float64(self.open_file.readline().split()[1])
        else if ".tif" in file:
            srcband = src_ds.GetRasterBand(1)
            GeoT = src_ds.GetGeoTransform()
            self.NODATA = srcband.GetNoDataValue()
            self.xllcorner = GeoT[0]
            self.yllcorner = GeoT[3]
            self.cellsize = srcband.GetScale()
            self.totalRows = srcband.YSize
            self.totalCols = srcband.XSize
  
    """
        opens file_name and sets it to open_file
    """
    def _openFile(self):
        if ".asc" in self.file_name:
            self.open_file=open(file_name, 'r')
        elif ".tif" in self.file_name:
            self.open_file=gdal.Open(self.file_name)

    """
        Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping..."
        exit(1)

    """
    """
    #def run(self):

    """
    """
    def _getLine(self, row):
        if ".asc" in self.file_name:
            f=self.open_file.readline()
        elif ".tif" in self.file_name:
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
        count = 0
        while count < self.totalRows:
             self.output_pipe.send(self.getLine(count))  
        count += 1

        print "entire file loaded"


