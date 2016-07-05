from multiprocessing import Process, Pipe
import numpy as np
from os.path import exists
from os import remove

"""
dataSaver

Class that saves data to a given input file and gets it from a Pipe object
designed to run as a separate process and inherits from Process module

currently supported output file types: ESRI ASCII format (.asc)]
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
        self.outFile = None
        self.fileName = outputFile 
        self.input_pipe=input_pipe

        #unpack header info
        self.totalCols = header[0]
        self.totalRows = header[1]
        self.cellsize = header[2]
        self.NODATA = header[3]
        self.xllcorner = header[4]
        self.yllcorner = header[5]

    """
    run

    calls functions needed to write all data to file_name
    """
    def run(self):
        self._openFile()
        self.write_func()
        self.outFile.close()

    """
    stop
    
    Alerts the thread that it needs to quit
    """
    def stop(self):
        print "Stopping..."
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

    """
    write_func

    takes data rows from input_pipe and writes them to outputFile
    writes exactly as many rows as defined in the header
    """
    def write_func(self):
        nrows = self.totalRows
        print nrows
        count = 0
        ln=""
        while nrows > 0:
            # get line from pipe
            try:
                arr=self.input_pipe.recv()
            except EOFError:
                print "Pipe empty"
                break
            
            for i in range(len(arr)):
                ln+=str(arr[i])
                if not i == len(arr) - 1:
                    ln+=' '
            ln+='\n'
            count+=1
            if count > 15:
                # write out accumulated lines
                self.outFile.write(ln)
                self.outFile.flush()
                count = 0
                ln=""
            nrows-=1
        self.outFile.write(ln)

        print "File completely written"

