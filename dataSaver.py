from multiprocessing import Process,Condition, Lock
import memoryInitializer
import numpy as np

from os.path import exists
from os import remove

class dataSaver(Process):
  
    def __init__(self, header, output_file, input_pipe):
        self.outFile = self.openFile(output_file)
        self.input_pipe=input_pipe
        #unpack header info
        self.totalCols = header[0]
        self.totalRows = header[1]
        self.cellsize = header[2]
        self.NODATA = header[3]
        self.xllcorner = header[4]
        self.yllcorner = header[5]

        outFile.write(header_str = ("ncols %f\n"
								"nrows %f\n"
								"xllcorner %f\n"
								"yllcorner %f\n"
								"cellsize %f\n"
								"NODATA_value %f"
								% (self.totalCols, self.totalRows, self.xllcorner, self.yllcorner, self.cellsize, self.NODATA)
								))

    def stop(self):
        print "Stopping..."
        exit(1)

    def openFile(self, fileName):
        if exists(fileName):
            print fileName, "already exists. Deleting it..."
            remove(fileName)

        try:
            self.outFile = open(fileName, 'w')
        except IOError:
            print "cannot open", fileName
            self.stop()
        except ValueError:
            print "Output file name was not a string"
            self.stop()

        self.outFile.write(self.header)

    def write_func(self):
        nrows = self.totalRows
        count = o
        ln=""
        while nrows > 0:
            # get line from pipe
            arr=self.input_pipe.recv()
            for i in arr:
                ln+=str(i)
            ln+='\n'
            count+=1
            if count > 15:
                # write out accumulated lines
                self.outFile.write(ln)
                self.outFile.flush()
                count = 0
                ln=""
            nrows-=count
        print "File completely written"

    def run(self, fileName):
        #self.openFile(fileName)
        self.write_func()
        self.outFile.close()

