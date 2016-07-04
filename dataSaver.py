from multiprocessing import Process, Pipe
import numpy as np
from os.path import exists
from os import remove

class dataSaver(Process):
  
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

    def run(self):
        self._openFile(self.fileName)
        self.write_func()
        self.outFile.close()

    def stop(self):
        print "Stopping..."
        exit(1)

    def _openFile(self, fileName):
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
                break
                print "Pipe empty"
            
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

        print "File completely written"

