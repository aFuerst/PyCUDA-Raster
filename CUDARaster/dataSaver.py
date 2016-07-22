from multiprocessing import Process, Pipe
import numpy as np
import os.path
import os
from osgeo import gdal
import Tkinter as tk
import ttk

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
        output_file - must be a valid file path as a string
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
                 Includes geotiff information if a tif input was used.
        file_type - a string which represents the file extension for input/output
        input_pipe - a Pipe object to read information from

    opens the output file and grabs the header information
    sets several instance variables
    """
    def __init__(self, _output_file,  header, _input_pipe):
        Process.__init__(self)
    
        if os.path.exists(os.path.realpath(__file__)[:-len("dataSaver.py")] + "data_saver_log.txt"):
            os.remove(os.path.realpath(__file__)[:-len("dataSaver.py")] + "data_saver_log.txt")
        self.logfile = open(os.path.realpath(__file__)[:-len("dataSaver.py")] + "data_saver_log.txt", 'w')

        self.file_name = _output_file 
        self.input_pipe = _input_pipe
        self.log("done init")

        #unpack header info
        self.totalCols = header[0]
        self.totalRows = header[1]
        self.cellsize = header[2]
        self.NODATA = header[3]
        self.xllcorner = header[4]
        self.yllcorner = header[5]
        self.GeoT = header[6]
        self.prj = header[7]
        self.log((header))

    def log(self, message):
        self.logfile.write(str(message) + '\n')
        print str(message)
        self.logfile.flush()

    def __del__(self):
        pass

    """
    run

    calls functions needed to write all data to file_name and render
    a progress bar
    """
    def run(self):
        self._openFile()
        self._gui()
        self._writeFunc()
        self._closeFile()

    """
    _gui

    tkinter gui to dispaly write out progress
    """
    def _gui(self):
        self.rt = tk.Tk()
        self.pb=ttk.Progressbar(mode="determinate", maximum=self.totalRows)
        self.lb = ttk.Label(text = self.file_name + " progress")
        self.lb.pack(side="top", fill="x")
        self.pb.pack(side="bottom", fill="x")

    """
    _closeFile

    
    """
    def _closeFile(self):
        self.dataset.FlushCache()


    """
    stop
    
    Alerts the thread that it needs to quit
    """
    def stop(self):
        self.log("Stopping saver " + self.file_name  +" ...")
        exit(1)

    """
    _openFile

    opens output_file and writes header information to it
    stores open file object in instance variable 
    """
    def _openFile(self):
        if exists(self.file_name):
            self.log(self.file_name + " already exists. Deleting it...")
            remove(self.file_name)
        self.driver = gdal.GetDriverByName('GTiff')
        self.dataset = self.driver.Create(self.file_name, self.totalCols, self.totalRows, 1, gdal.GDT_Float32)
        self.dataset.GetRasterBand(1).SetNoDataValue(self.NODATA)
        self.dataset.SetGeoTransform(self.GeoT)
        try:
            self.dataset.SetProjection(str(self.prj))
        except RuntimeError:
            self.dataset.SetProjection('')

    """
    _writeFunc

    takes data rows from input_pipe and writes them to output_file
    writes exactly as many rows as defined in the header
    """
    def _writeFunc(self):
        nrows = 0
        self.guiMade.wait()
        while nrows < self.totalRows:
            # get line from pipe
            try:
                arr=self.input_pipe.recv()
            except EOFError:
                self.log("Pipe closed unexpectedly")
                self.stop()
            self.dataset.GetRasterBand(1).WriteArray(np.float32([arr]), 0, nrows)
            if nrows % 50 == 0:
                self.dataset.FlushCache()
            nrows+=1
            self.pb.step(1)
            self.rt.update()
        self.log( "Output %s written to disk" % self.file_name)

