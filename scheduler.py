
import dataLoader, gpuCalc, dataSaver
import numpy as np

from multiprocessing import Process, Pipe, active_children
from time import sleep

"""
scheduler.py
Starts and manages processes which load data, do raster calculations on GPU,
and save data back to disk.

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

#NOTE: USAGE: scheduler.py input output_1 func_1 output_2 func_2 ... output_n func_n

# input and output files must have the same file type

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
        if loader.exitcode != None and loader.exitcode != 0:
            print "Error encountered in data loader, ending tasks"            
            calc.stop()
            for saver in savers:
                saver.stop()
            break
        if calc.exitcode != None and calc.exitcode != 0:
            loader.stop()
            for saver in savers:
                saver.stop()
            print "Error encountered in GPU calculater, ending tasks"
            break
        sleep(1)

if __name__ == '__main__':
    #If run from the command line, parse arguments.
    from sys import argv
    outFiles = []
    funcs = []
    for i in range(2,len(argv),2):
        outFiles.append(argv[i])
        funcs.append(argv[i+1].lower())
    run(argv[1], outFiles, funcs)
