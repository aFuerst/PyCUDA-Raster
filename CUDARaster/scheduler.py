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
        logfile.write("made pipes\n")
        logfile.flush()
        print type(inputFile)

        if type(inputFile) is unicode or type(inputFile) is str:
            logfile.write("loading from disk\n")
            logfile.flush()
            print "loading from disk"
            loader = dataLoader.dataLoader(inputFile, inputPipe[0])
        else:
            print "loading from qgis"
            logfile.write("loading from qgis\n")
            logfile.flush()
            loader = layerLoader.layerLoader(inputFile, inputPipe[0])

        logfile.write("made loader\n")    
        logfile.flush()
        header = loader.getHeaderInfo()
        print header
        logfile.write(str(header) + "\n")

        logfile.write("got header info\n")
        logfile.flush()
        calc = gpuCalc.GPUCalculator(header, inputPipe[1], map((lambda x: x[0]), outputPipes), functions)
        logfile.write("made gpu calc\n")

        savers = []
        for i in range(len(outputFiles)):
            savers.append(dataSaver.dataSaver(outputFiles[i], header, outputPipes[i][1]))
        logfile.write("made saver threads\n")
        logfile.flush()

        # start all threads
        loader.start()
        logfile.write("started loader\n")
        logfile.flush()

        calc.start()
        logfile.write("started gpuCalc\n")
        logfile.flush()

        for i in range(len(outputFiles)):
            savers[i].start()
        logfile.write("started savers\n")

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
    disk_rows = 15  # 30 appears to be optimal number of rows to read at a time for any file
    for i in range(2,len(argv), 2):
        outFiles.append(argv[i])
        funcs.append(argv[i+1].lower())
    run(argv[1], outFiles, funcs, disk_rows)

