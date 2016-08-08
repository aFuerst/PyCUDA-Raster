from PyQt4.QtCore import *
from PyQt4.QtCore import *

import qgis
from qgis.core import *
from qgis.gui import *
from qgis.utils import iface

from math import isnan
from numpy import float32, float64

from multiprocessing import Process, Pipe
import struct, os, os.path
from time import time

"""
layerLoader
Loader which takes information from an active QGIS layer

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

class layerLoader(Process):

    def __init__(self, inputLayer, output_pipe):
        Process.__init__(self)

        if os.path.exists(os.path.realpath(__file__)[:-len("layerLoader.py")] + "layer_stuff_log.txt"):
            os.remove(os.path.realpath(__file__)[:-len("layerLoader.py")] + "layer_stuff_log.txt")
        self.logfile = open(os.path.realpath(__file__)[:-len("layerLoader.py")] + "layer_stuff_log.txt", 'w')

        self.log("init")
        self.output_pipe = output_pipe
        self.layer = inputLayer
        self.log(str(self.layer))
        self.log("grabbed layer")
        self.ext = self.layer.extent()
        self.d = self.layer.dataProvider().block(1, self.ext, self.layer.width(), self.layer.height())
        self.log("done init")
        self._readHeaderInfo()
        self.log("returning from init")

    def _readHeaderInfo(self):
        self.log( "reading header info")
        self.totalCols = self.layer.width()        
        self.totalRows = self.layer.height()
        self.log("doing corners")
        self.xllcorner = self.layer.extent().xMinimum()
        self.yllcorner = self.layer.extent().yMinimum()
        self.log("doing cellsize")
        self.cellsize = self.layer.rasterUnitsPerPixelY()
        self.log("doing NODATA")
        s = self.layer.metadata()
        loc = s.find("No Data Value</p>\n<p>") + len("No Data Value</p>\n<p>")
        loc2 = s.find("<", loc)
        self.NODATA = float64(s[loc:loc2])
        self.prj = self.layer.crs().toWkt()
        self.GeoT = (self.xllcorner, self.layer.rasterUnitsPerPixelX(), 0, self.yllcorner, 0, self.layer.rasterUnitsPerPixelY())
        self.log("done header info")
        self.log("header in layerLoader: " + str((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj)))

    def run(self):
        self.log("starting run")
        self._loadFunc()

    def _getLine(self, row):
        arr = []
        for x in range(self.layer.width()):
            if isnan(self.d.value(row,x)): 
                self.log("illegal value, breaking, (" + str(x) + ", " + str(row) + ")")
                self.stop()
            arr.append(self.d.value(row,x))
        return float32(arr)

    def getFileType(self):
        return "tif" 

    """
    stop 

    Alerts the thread that it needs to quit
    """
    def stop(self):
        self.log("Stopping layer stuff...")
        exit(1)

    """
    _loadFunc

    sends data one row at a time to output_pipe, sends exactly the number of rows as are in the input file
    """
    def _loadFunc(self):
        # must go from totalRows to 0 because QGIS is giving us te data backwards because it is stupid.
        count = self.totalRows - 1
        self.log("in load")
        while count >= 0:
            self.output_pipe.send(self._getLine(count))
            count -= 1
        self.output_pipe.close()
        self.log( "Input file loaded from disk")

    def getHeaderInfo(self):
        self.log((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj))
        return (self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj)

    def log(self, message):
        self.logfile.write(str(message) + '\n')
        print str(message)
        self.logfile.flush()

