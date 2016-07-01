from multiprocessing import Process,Condition, Lock
import memoryInitializer
import numpy as np

from os.path import exists
from os import remove

class dataSaver(Process):
  
    def __init__(self, _memInit, _header=None):
        self.memInit = _memInit
        self.header = _header
        self.outFile = None

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
            self.stop

        if self.header == None:
            print "Getting header information from memInit not yet implemented."
            print "Exiting..."
            self.stop()
        else:
            self.outFile.write(self.header)

    def write_func(self):
        nrows = self.memInit.totalRows

        while nrows > 0:
            self.memInit.from_gpu_buffer_lock.acquire()
            while not self.memInit.from_gpu_buffer_full.is_set():
                self.memInit.from_gpu_buffer_lock.wait()

            ln=""
            count=0
            #Want to ignore buffer rows
            for row in range(1, len(self.memInit.from_gpu_buffer)-1):
                for col in self.memInit.from_gpu_buffer[row]:
                    ln+=str(col)
                    ln+=' '
                ln+='\n'
                count+=1
                if count > 15:
                    count=0
                    f.write(ln)
                    ln=""
            
            if count <= 15:
                f.write(ln)

            f.flush()  
            nrows-=len(self.memInit.from_gpu_buffer)

            self.memInit.from_gpu_buffer_full.clear()
            self.memInit.from_gpu_buffer_lock.notify()
            self.memInit.from_gpu_buffer_lock.release()

    def run(self, fileName):
        self.openFile(fileName)
        self.write_func()
        self.outFile.close()
