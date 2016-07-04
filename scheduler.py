import dataLoader, gpuCalc, dataSaver
import numpy as np
from multiprocessing import Process, Pipe, Connection

def run(inputFile, outputFile):
    # create input and output pipes    
    inputPipe = Pipe()
    outputPipe = Pipe()

    loader = dataLoader(inputFile, inputPipe[0])
    header = loader.getHeaderInfo()
    calc = GPUCalculator(header, inputPipe[1], outputPipe[0])
    saver = dataSaver(outputFile, outputPipe[1])

    # start all threads
    loader.start()
    calc.start()
    saver.start()
    
    # join all threads
    loader.join()
    calc.join()
    saver.join()

if __name__ == '__main__':
    run("aigrid.asc", "output.asc")
