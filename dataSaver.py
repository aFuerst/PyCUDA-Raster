<<<<<<< HEAD
from multiprocessing import Process, Pipe
import numpy as np
from os.path import exists
from os import remove
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

copyright            : (C) 2016 by Alex Feurst, Charles Kazer, William Hoffman
email                : fuersta1@xavier.edu, ckazer1@swarthmore.edu, whoffman1@gulls.salisbury.edu

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
class dataSaver(Process):
  
    """
    __init__

    paramaters:
        output_file - must be a valid file path as a string
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
                 Includes geotiff information if a tif input was used.
        input_pipe - a Pipe object to read information from

    opens the output file and grabs the header information
    sets several instance variables
    """
    def __init__(self, _output_file,  header, _input_pipe, _disk_rows):
        Process.__init__(self)
    
        self.file_name = _output_file 
        self.input_pipe = _input_pipe
        self.write_rows = _disk_rows
        self.header = header

    #--------------------------------------------------------------------------#

    def __del__(self):
        pass

    #--------------------------------------------------------------------------#

    """
    run

    calls functions needed to write all data to file_name and render
    a progress bar
    """
    def run(self):
        self._unpackHeader()
        self._openFile()
        self._gui()
        self._writeFunc()
        self._closeFile()

    #--------------------------------------------------------------------------#
    
    """
    _unpackHeader
    
    Puts header info into class vars
    """
    def _unpackHeader(self):
        #unpack header info
        self.totalCols = self.header[0]
        self.totalRows = self.header[1]
        self.cellsize = self.header[2]
        self.NODATA = self.header[3]
        self.xllcorner = self.header[4]
        self.yllcorner = self.header[5]
        self.GeoT = self.header[6]
        self.prj = self.header[7]   

    #--------------------------------------------------------------------------# 
 
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

    #--------------------------------------------------------------------------#

    """
    _closeFile

    writes out any remaining data, closes GDAL file
    """
    def _closeFile(self):
        self.dataset.FlushCache()
        self.dataset = None

    #--------------------------------------------------------------------------#

    """
    stop
    
    Alerts the thread that it needs to quit
    Closes file and pipes
    """
    def stop(self):
        print "Stopping saver ", self.file_name ," ..."
        self._closeFile()
        self.input_pipe.close()
        exit(1)

    #--------------------------------------------------------------------------#

    """
    _openFile

    opens output_file and writes header information to it
    stores open file object in instance variable 
    """
    def _openFile(self):
        if exists(self.file_name):
            print self.file_name, "already exists. Deleting it..."
            remove(self.file_name)
        self.driver = gdal.GetDriverByName('GTiff')
        self.dataset = self.driver.Create(self.file_name, self.totalCols, self.totalRows, 1, gdal.GDT_Float32, options = ['COMPRESS=DEFLATE', 'NUM_THREADS=2', 'BIGTIFF=YES'])
        self.dataset.GetRasterBand(1).SetNoDataValue(self.NODATA)
        self.dataset.SetGeoTransform(self.GeoT)
        try:
            self.dataset.SetProjection(str(self.prj))
        except RuntimeError:
            print "Warning: Invalid projection."
            self.dataset.SetProjection('')

    #--------------------------------------------------------------------------#

    """
    _writeFunc

    takes data rows from input_pipe and writes them to output_file
    writes exactly as many rows as defined in the header
    """
    def _writeFunc(self):
        nrows = 0
        arr = np.ndarray(shape=(self.write_rows, self.totalCols), dtype=np.float32)
        np_write_arr = [i for i in range(self.totalCols)]
        while nrows < self.totalRows:
            # remaining rows < write_rows, only write in as many as are extra
            if nrows + self.write_rows >= self.totalRows:
                rem = self.totalRows - nrows
                arr.resize((rem, self.totalCols))
                try:
                    for row in range(rem):
                        np.put(arr[row], np_write_arr, self.input_pipe.recv())
                except EOFError:
                    print "Pipe closed unexpectedly"
                    self.stop()
            else:
                # write in as many rows as write_rows indicates
                try:
                    for row in range(self.write_rows):
                        np.put(arr[row], np_write_arr, self.input_pipe.recv())
                except EOFError:
                    print "Pipe closed unexpectedly"
                    self.stop()
            # write out rows
            self.dataset.GetRasterBand(1).WriteArray(arr, 0, nrows)
            nrows+=self.write_rows
            self.pb.step(self.write_rows)
            self.rt.update()
        # write out remaining lines
        self.dataset.FlushCache()
        print "Output %s written to disk" % self.file_name

if __name__=="__main__":
    pass

