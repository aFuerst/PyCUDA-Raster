from PyQt4.QtCore import *
from PyQt4.QtCore import *

import qgis
from qgis.core import *
from qgis.gui import *
from qgis.utils import iface

from math import isnan
import numpy as np

from multiprocessing import Process, Pipe
import struct, os, os.path
from time import time

class layerStuff(Process):

    def __init__(self, inputLayer, output_pipe):
        Process.__init__(self)

        if os.path.exists(os.path.realpath(__file__)[:-len("layerStuff.py")] + "layer_stuff_log.txt"):
            os.remove(os.path.realpath(__file__)[:-len("layerStuff.py")] + "layer_stuff_log.txt")
        self.logfile = open(os.path.realpath(__file__)[:-len("layerStuff.py")] + "layer_stuff_log.txt", 'w')
        #self.log( os.path.realpath(__file__)[:-14] + "layer_stuff_log.txt")

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
        #rlayer.rasterUnitsPerPixelX() * 
        s = self.layer.metadata()
        loc = s.find("No Data Value</p>\n<p>") + len("No Data Value</p>\n<p>")
        loc2 = s.find("<", loc)
        self.NODATA = np.float64(s[loc:loc2])
        self.prj = self.layer.crs().authid()
        self.GeoT = (self.xllcorner, self.layer.rasterUnitsPerPixelX(), 0, self.yllcorner, 0, self.layer.rasterUnitsPerPixelY())
        self.log("done header info")
        self.log("header in layerStuff: " + str((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj)))

    def run(self):
        self.log("starting run")
        self._loadFunc()

    def _getLine(self, row):
        #self.log("in getline")
        #for y in range(rlayer.height()):
        arr = []
        #self.log("fetching row: " + str(row))
        for x in range(self.layer.width()):
            if isnan(self.d.value(row,x)): 
                self.log("illegal value, breaking, (" + str(x) + ", " + str(row) + ")")
                break
            arr.append( self.d.value(row,x))
        return np.float64(arr)

    def getFileType(self):
        return "tif" 

    """
    _loadFunc

    sends data one row at a time to output_pipe, sends exactly the number of rows as are in the input file
    """
    def _loadFunc(self):
        count = 0
        self.log("in load")
        while count < self.totalRows:
            #self.log( "sending line")
            self.output_pipe.send(self._getLine(count))
            count += 1
        self.output_pipe.close()
        self.log( "Input file loaded from disk")

    def getHeaderInfo(self):
        self.log((self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj))
        return (self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj)

    def log(self, message):
        self.logfile.write(str(message) + '\n')
        print str(message)
        self.logfile.flush()

"""
    #layerpath = "/home/afuerst1/Documents/Lembo-REU-2016/aigrid.tif"
    fileInfo = QFileInfo(layerpath)
    baseName = fileInfo.baseName()

    rlayer = QgsRasterLayer(layerpath, baseName)
    rlayer.width(), rlayer.height()

    ext = rlayer.extent()

    d = rlayer.dataProvider().block(1, ext, rlayer.width(), rlayer.height())


        
        
    QgsMapLayerRegistry.instance().addMapLayer(rlayer)
"""

