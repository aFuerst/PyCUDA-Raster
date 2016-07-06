import dataLoader, gpuCalc, dataSaver
import numpy as np

from multiprocessing import Process, Pipe, active_children
from sys import argv
from time import sleep

#NOTE: USAGE: scheduler.py input output_1 func_1 output_2 func_2 ... output_n func_n

def run(inputFile, outputFiles, functions):
    
    # create input and output pipes    
    inputPipe = Pipe()
    outputPipes = []
    for i in range(len(outputFiles)):
        outputPipes.append(Pipe())

    loader = dataLoader.dataLoader(inputFile, inputPipe[0])
    header = loader.getHeaderInfo()
    calc = gpuCalc.GPUCalculator(header, inputPipe[1], map((lambda x: x[0]), outputPipes), functions)
    savers = []
    for i in range(len(outputFiles)):
        savers.append(dataSaver.dataSaver(outputFiles[i], header, outputPipes[i][1]))

    # start all threads
    loader.start()
    calc.start()
    for i in range(len(outputFiles)):
        savers[i].start()
    
    # join all threads
    while active_children():
        sleep(5)
    #loader.join()
    #calc.join()
    #saver.join()

if __name__ == '__main__':
    outFiles = []
    funcs = []
    for i in range(2,len(argv),2):
        outFiles.append(argv[i])
        funcs.append(argv[i+1])
    run(argv[1], outFiles, funcs)
