from osgeo import gdal
from multiprocessing import Process, Pipe
import struct, os
import numpy as np

gdal.UseExceptions()
fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

"""
dataLoader

Class that reads data from a given input file and passes it to a Pipe object
designed to run as a separate process and inherits from Process module

currently supported input file types: [GEOTiff (.tif), ESRI ASCII format (.asc)]

ERROR: GEOTiff files do NOT work, they are corrupting at some point 
"""
class dataLoader(Process):

    """
    __init__

    paramaters:
        inputFile - must be a valid file path as a string
        output_pipe - a Pipe object to pass read information into
   
    opens the input file and grabs the header information
    sets several instance variables
    """
    def __init__(self, inputFile, output_pipe):
        Process.__init__(self)
        self.output_pipe = output_pipe
        self.file_name = inputFile
        self._openFile()
        self._readHeaderInfo()
        self.cur_line=""
        self.prev_last_row=""

    """
    getHeaderInfo
    
    Returns header information as a six-tuple in this order:
    (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    """
    def getHeaderInfo(self):
        return self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner

    """
    _readHeaderInfo

    requires file to be opened already, gets header info from file and puts it in instance variables
    """
    def _readHeaderInfo(self):
        if ".asc" in self.file_name:
            self.totalCols = np.int64(self.open_file.readline().split()[1])
            self.totalRows = np.int64(self.open_file.readline().split()[1])
            self.xllcorner = np.float64(self.open_file.readline().split()[1])
            self.yllcorner = np.float64(self.open_file.readline().split()[1])
            self.cellsize = np.float64(self.open_file.readline().split()[1])
            self.NODATA = np.float64(self.open_file.readline().split()[1])
        elif ".tif" in self.file_name:
            src_ds = gdal.Open(self.file_name) #open again to get GeoT info
            srcband = self.open_file
            print "here3"
            GeoT = src_ds.GetGeoTransform()
            print "here4"
            # SEGMENTATION FAULT HERE
            self.NODATA = srcband.GetNoDataValue()
            print "here"
            self.xllcorner = GeoT[0]
            self.yllcorner = GeoT[3]
            src_ds = None
            print "here2"
            self.cellsize = self.open_file.GetScale()
            self.totalRows = self.open_file.YSize
            self.totalCols = self.open_file.XSize
  
    """
    _openFile

    opens file_name and sets it to open_file, supports '.tif' and '.asc' files
    """
    def _openFile(self):
        if ".asc" in self.file_name:
            self.open_file=open(self.file_name, 'r')
        elif ".tif" in self.file_name:
            f = gdal.Open(self.file_name)
            self.open_file = f.GetRasterBand(1)

    """
    stop 

    Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping..."
        exit(1)

    """
    run

    calls functions needed to read all data from file_name
    """
    def run(self):
        self._loadFunc()

    """
    _getLine

    returns a single row from the open file as a numpy float64 array
    """
    def _getLine(self, row):
        if ".asc" in self.file_name:
            f=self.open_file.readline().split()
        elif ".tif" in self.file_name:
            try:
                # ERROR
                # GDAL Raster File gets corrupted between opening and here
                # DOES NOT WORK
                print self.open_file.DataType
                print gdal.GetDataTypeName(self.open_file.DataType)
                f=struct.unpack(fmttypes[gdal.GetDataTypeName(self.open_file.DataType)]*self.totalCols, srcband.ReadRaster(0,row,srcband.XSize,1, buf_type=self.open_file.DataType))
            except RuntimeError:
                f=[]   
        return np.float64(f)

    """
    _loadFunc

    sends data one row at a time to output_pipe, sends exactly the number of rows as are in the input file
    """
    def _loadFunc(self):
        count = 0
        while count < self.totalRows:
            self.output_pipe.send(self._getLine(count))
            count += 1
        self.output_pipe.close()
        print "Input file loaded from disk"


