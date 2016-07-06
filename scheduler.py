import dataLoader, gpuCalc, dataSaver
import numpy as np

from multiprocessing import Process, Pipe
from sys import argv

def run(inputFile, outputFile, functionName, multArgs=False):
    
    if multArgs:
        print "Not implemented"
        exit()

    else:
        # create input and output pipes    
        inputPipe = Pipe()
        outputPipe = Pipe()

        loader = dataLoader.dataLoader(inputFile, inputPipe[0])
        header = loader.getHeaderInfo()
        calc = gpuCalc.GPUCalculator(header, inputPipe[1], outputPipe[0], functionName)
        saver = dataSaver.dataSaver(outputFile, header, outputPipe[1])

        # start all threads
        loader.start()
        calc.start()
        saver.start()
        
        # join all threads
        loader.join()
        calc.join()
        saver.join()

if __name__ == '__main__':
    if argv[1] == '-multi':
        outFiles = []
        funcs = []
        for i in range(3,len(argv),2):
            outFiles.append(argv[i])
            funcs.append(argv[i+1])
        run(argv[1], outFiles, funcs, True)
    else:
        run(argv[1], argv[2], argv[3])
