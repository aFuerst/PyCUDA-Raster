from multiprocessing import Process, Pipe
import numpy as np
from os.path import exists
from os import remove
from osgeo import gdal

gdal.UseExceptions()
fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

"""
dataSaver

Class that saves data to a given input file and gets it from a Pipe object
designed to run as a separate process and inherits from Process module

currently supported output file types: ESRI ASCII format (.asc), GEOTiff (.tif)
"""
class dataSaver(Process):
  
    """
    __init__

    paramaters:
        outputFile - must be a valid file path as a string
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
        input_pipe - a Pipe object to read information from

    opens the output file and grabs the header information
    sets several instance variables
    """
    def __init__(self, outputFile,  header, input_pipe):
        Process.__init__(self)
        #self.outFile = None
        self.fileName = outputFile 
        self.input_pipe=input_pipe

        #unpack header info
        self.totalCols = header[0]
        self.totalRows = header[1]
        self.cellsize = header[2]
        self.NODATA = header[3]
        self.xllcorner = header[4]
        self.yllcorner = header[5]
        if ".tif" in self.fileName:
            self.GeoT = header[6]
            self.prj = header[7]

    """
    run

    calls functions needed to write all data to file_name
    """
    def run(self):
        self._openFile()
        self.write_func()
        self._closeFile()

    """
    """
    def _closeFile(self):
        if ".tif" in self.fileName:
            self.dataset.FlushCache()
        elif ".asc" in self.fileName:
            self.outFile.close()
        pass
        
    """
    stop
    
    Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping saver..."
        exit(1)

    """
    _openFile

    opens outputFile and writes header information to it
    stores open file object in instance variable 
    """
    def _openFile(self):
        if exists(self.fileName):
            print self.fileName, "already exists. Deleting it..."
            remove(self.fileName)
        if ".asc" in self.fileName:
            try:
                self.outFile = open(self.fileName, 'w')
            except IOError:
                print "cannot open", self.fileName
                self.stop()
            except ValueError:
                print "Output file name was not a string"
                self.stop()

            # write out header
            self.outFile.write(
                    "ncols %f\n"
                    "nrows %f\n"
                    "xllcorner %f\n"
                    "yllcorner %f\n"
                    "cellsize %f\n"
                    "NODATA_value %f\n"
                    % (self.totalCols, self.totalRows, self.xllcorner, self.yllcorner, self.cellsize, self.NODATA)
                    )
        elif ".tif" in self.fileName:
            #y_pixels = self.totalRows  # number of pixels in x
            #x_pixels = self.totalCols  # number of pixels in y
            #print x_pixels, y_pixels
            #PIXEL_SIZE = self.cellsize  # size of the pixel...        
            #x_min = self.xllcorner  
            #y_max = self.yllcorner  # x_min & y_max are like the "top left" corner.
            self.driver = gdal.GetDriverByName('GTiff')
            self.dataset = self.driver.Create(self.fileName, self.totalCols, self.totalRows, 1, gdal.GDT_Float32)
            self.dataset.GetRasterBand(1).SetNoDataValue(self.NODATA)
            self.dataset.SetGeoTransform(self.GeoT)
            self.dataset.SetProjection(self.prj)

    """
    write_func

    takes data rows from input_pipe and writes them to outputFile
    writes exactly as many rows as defined in the header
    """
    def write_func(self):
        nrows = 0
        while nrows < self.totalRows - 1:
            #print nrows
            # get line from pipe
            try:
                arr=self.input_pipe.recv()
            except EOFError:
                print "Pipe empty"
                return
            if ".asc" in self.fileName:            
                arr.tofile(self.outFile, sep=" ", format="%f")
                self.outFile.write('\n')
            elif ".tif" in self.fileName:
                arr = np.float32([arr])
                self.dataset.GetRasterBand(1).WriteArray(arr, 0, nrows)
                if nrows % 50 == 0:
                    self.dataset.FlushCache()
            nrows+=1
        print "Output %s written to disk" % self.fileName

