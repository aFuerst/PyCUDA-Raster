from osgeo import gdal
from multiprocessing import Process, Pipe
import struct, os, os.path
import numpy as np

gdal.UseExceptions()
fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

"""
dataLoader

Class that reads data from a given input file and passes it to a Pipe object
designed to run as a separate process and inherits from Process module

currently supported input file types: [GEOTiff (.tif), ESRI ASCII format (.asc)]
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
    def __init__(self, _input_file, _output_pipe):
        Process.__init__(self)
        print os.path.realpath(__file__)
        if os.path.exists(os.path.realpath(__file__)[:-len("dataLoader.py")+1] + "data_loader_log.txt"):
            os.remove(os.path.realpath(__file__)[:-len("dataLoader.py")+1] + "data_loader_log.txt")
        self.logfile = open(os.path.realpath(__file__)[:-len("dataLoader.py")+1] + "data_loader_log.txt", 'w')

        self.output_pipe = _output_pipe
        self.file_name = _input_file
        self.file_type = None
        self._openFile()
        self._readHeaderInfo()
        self.log("init done")

    def log(self, message):
        self.logfile.write(str(message) + '\n')
        print str(message)
        self.logfile.flush()

    """
    getHeaderInfo
    
    Returns header information as a six-tuple in this order:
    (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner, GeoT, prj)
    """
    def getHeaderInfo(self):
        self.log("sending header")
        if "asc" == self.file_type:
            self.log( (self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner))
            return self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner
        elif "tif" == self.file_type:
            self.log( (self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj))
            return self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj

    """
    _readHeaderInfo

    requires file to be opened already, gets header info from file and puts it in instance variables
    """
    def _readHeaderInfo(self):
        self.log("reading header info")
        if ".asc" in self.file_name:
            self.totalCols = np.int64(float(self.open_file.readline().split()[1]))
            self.totalRows = np.int64(float(self.open_file.readline().split()[1]))
            self.xllcorner = np.float64(self.open_file.readline().split()[1])
            self.yllcorner = np.float64(self.open_file.readline().split()[1])
            self.cellsize = np.float64(self.open_file.readline().split()[1])
            self.NODATA = np.float64(self.open_file.readline().split()[1])

            self.log((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner))
        elif ".tif" in self.file_name:
            self.GeoT = self.open_file.GetGeoTransform()
            self.prj = self.open_file.GetProjection()
            self.NODATA = self.open_raster_band.GetNoDataValue()
            self.xllcorner = self.GeoT[0]
            self.yllcorner = self.GeoT[3]
            self.cellsize = self.open_raster_band.GetScale()
            self.totalRows = self.open_raster_band.YSize
            self.totalCols = self.open_raster_band.XSize
            self.log((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj))

    """
    _openFile

    opens file_name and sets it to open_file, supports '.tif' and '.asc' files
    """
    def _openFile(self):
        self.log("opening read file")
        if ".asc" in self.file_name:
            self.open_file=open(self.file_name, 'r')
            self.file_type = "asc"
        elif ".tif" in self.file_name:
            self.open_file = gdal.Open(self.file_name)
            self.open_raster_band = self.open_file.GetRasterBand(1)
            self.dataType = self.open_raster_band.DataType
            self.unpackVal = fmttypes[gdal.GetDataTypeName(self.dataType)]*self.open_raster_band.XSize
            self.file_type = "tif"
        else:
            print "Unsupported file type"
            self.stop()
        self.log("read file open")

    """
    stop 

    Alerts the thread that it needs to quit
    """
    def stop(self):
        self.log("Stopping loader...")
        exit(1)

    """
    run

    calls functions needed to read all data from file_name
    """
    def run(self):
        self._loadFunc()

    def getFileType(self):
        return self.file_type

    """
    _getLines

    Each _getLines function reads in a file of a different type and sends data
    to the GPUCalculator class
    """

    def _getLinesASC(self):
        for line in self.open_file:
            self.output_pipe.send(np.float64(line.split()))

    def _getLinesTIF(self):
        count = 0
        while count < self.totalRows:
            try:
                f=struct.unpack(self.unpackVal, self.open_raster_band.ReadRaster(0,count,self.totalCols,1, buf_type=self.dataType))
                self.output_pipe.send(np.float64(f))
            # EOF
            except RuntimeError:
                f=[]   
                return
            count += 1

    """
    _loadFunc

    sends data one row at a time to output_pipe, sends exactly the number of rows as are in the input file
    """
    def _loadFunc(self):
        if "asc" == self.file_type:
            self._getLinesASC()
        elif "tif" == self.file_type:
            self._getLinesTIF()
        self.output_pipe.close()
        self.log("Input file loaded from disk")

