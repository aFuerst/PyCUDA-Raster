import dataLoader, gpuCalc, dataSaver
import numpy as np

from multiprocessing import Process, Pipe, active_children
from time import sleep, time
import os.path, os

try:
    # allows to run without qgis
    import qgis, layerLoader
    from qgis.core import *
except ImportError:
    pass

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

def run(inputFile, outputFiles, functions, disk_rows = 20):
    print os.path.realpath(__file__)
    if os.path.exists(os.path.realpath(__file__)[:-len("scheduler.py")+1] + "scheduler_log.txt"):
        os.remove(os.path.realpath(__file__)[:-len("scheduler.py")+1] + "scheduler_log.txt")
    logfile = open(os.path.realpath(__file__)[:-len("scheduler.py")+1] + "scheduler_log.txt", 'w')
    logfile.write(str(inputFile))
    logfile.flush()

    print inputFile, "\nin scheduler\n"
    start = time()
    try:
        # create input and output pipes    
        inputPipe = Pipe()
        outputPipes = []
        for i in range(len(outputFiles)):
            outputPipes.append(Pipe())

        loader = dataLoader.dataLoader(inputFile, inputPipe[0], disk_rows)
        loader.start()
        header = loader.getHeaderInfo()

        calc = gpuCalc.GPUCalculator(header, inputPipe[1], map((lambda x: x[0]), outputPipes), functions)
        calc.start()
        
        savers = []
        for i in range(len(outputFiles)):
            savers.append(dataSaver.dataSaver(outputFiles[i], header, outputPipes[i][1], disk_rows))

        # start all threads
        for i in range(len(outputFiles)):
            savers[i].start()

        # join all threads
        while active_children():
            if loader.exitcode != None and loader.exitcode != 0:
                logfile.write("Error encountered in data loader, ending tasks\n")            
                calc.stop()
                for saver in savers:
                    saver.stop()
                return True
                break
            if calc.exitcode != None and calc.exitcode != 0:
                loader.stop()
                for saver in savers:
                    saver.stop()
                return True
                logfile.write("Error encountered in GPU calculater, ending tasks\n")
                break
            sleep(1)
    
    except IOError as e:
        logfile.write(str(e) + "\n")
        print e
        return True

    comp = time() - start
    print "Processing completed in: %d mins, %d secs" % (comp / 60, comp % 60)
    logfile.write("Processing completed in: %d mins, %d secs\n" % (comp / 60, comp % 60))
    logfile.write("program ended")
    return False


if __name__ == '__main__':
    from sys import argv
    outFiles = []
    funcs = []
    disk_rows = 50  
    for i in range(2,len(argv), 2):
        outFiles.append(argv[i])
        funcs.append(argv[i+1].lower())
    run(argv[1], outFiles, funcs, disk_rows)

