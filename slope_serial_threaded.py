import numpy as np
import threading
from multiprocessing import Process, Queue, Condition
from collections import deque
from time import sleep

"""
Threaded serial slope calculator for raster ascii files.
Uses producer-consumer paradigm to handle memory.

Charlie Kazer
"""

def main():

	input_file = open("aigrid.asc", 'r')
	ncols = input_file.readline().split()[1]
	nrows = input_file.readline().split()[1]
	xllcorner = input_file.readline().split()[1]
	yllcorner = input_file.readline().split()[1]
	cellsize = float(input_file.readline().split()[1])
	NODATA = float(input_file.readline().split()[1])


        #Create data buffer, conditions. Choose buffer size appropriate for memory size.
        data_buffer = Queue(maxsize=10000)

        output_file = open("output_slope.asc", 'w')
        
        #create threads
        load_proc = threading.Thread(target=load_func, args=(input_file, data_buffer))
        calc_proc = threading.Thread(target=calc_func, args=(output_file, data_buffer, int(nrows), int(ncols), float(cellsize), float(NODATA)))

        #set up header
        header_str = ("ncols %s\n"
                  "nrows %s\n"
                  "xllcorner %s\n"
                  "yllcorner %s\n"
                  "cellsize %f\n"
                  "NODATA_value %d\n"
                  % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
                 )

        output_file.write(header_str)

        #spin up header
        load_proc.start()
        calc_proc.start()

        calc_proc.join()

        #clean-up
        input_file.close()
        output_file.close()


#////////////////////////////////////////////////////////////////////////////////#

# load_func: Produces data from input file
def load_func(input_file, data_buffer):
  for line in input_file:
    # NOTE: Don't skip any lines here, the file pointer has already advanced
    # past the header to the data.
    while (data_buffer.full()):
      #So inefficient... But in order to get proper waiting, would need to introduce lock
      sleep(0.1)
      #print "loader waiting..."
      
    data_buffer.put(np.fromstring(line, sep=' '))


#////////////////////////////////////////////////////////////////////////////////#

# calc_func: Consumes data from shared buffer. Calls calc_slope to handle actual calculations
# Writes results to output file.
def calc_func(output_file, data_buffer, numRows, numCols, cellsize, NODATA):
  cur_lines = deque([], 3)
  count = 0
  cur_slope = []

  cur_lines.append(np.zeros(numCols).fill(NODATA))
  #Read first two lines so that when we enter main while, cur_lines will
  #always contain 3 lines
  while(len(cur_lines) < 2):
    while (data_buffer.empty()):
      sleep(0.1)
    cur_lines.append(data_buffer.get())
    count += 1

  #Calculate slope for top line
  for i in range(cur_lines[1].size):
    cur_slope.append(calc_slope(cur_lines, i, cellsize, NODATA))
  output_file.write(' '.join(cur_slope))
  output_file.write('\n')
  cur_slope = []

  #Main loop
  while (count < numRows):
    while (data_buffer.empty()):
      sleep(0.1)
      #print "waiting consumer..."

    cur_lines.append(data_buffer.get())
    count += 1

    for i in range(cur_lines[1].size):
      cur_slope.append(calc_slope(cur_lines, i, cellsize, NODATA))
    output_file.write(' '.join(cur_slope))
    output_file.write('\n')
    cur_slope = []

  #Calculate slope for bottom line
  cur_lines.append(np.zeros(numCols).fill(NODATA))
  for i in range(cur_lines[1].size):
      cur_slope.append(calc_slope(cur_lines, i, cellsize, NODATA))
  output_file.write(' '.join(cur_slope))
  output_file.write('\n')
  cur_slope = []



#////////////////////////////////////////////////////////////////////////////////#

# calc_slope: calculates the slope values for a cell of data
def calc_slope(cur_lines, col, cellsize, NODATA):
  if cur_lines[1][col] == NODATA:
    return str(NODATA)

  nbhd = [] #'neighborhood' of cell

  for i in range(3):
    for j in range(-1,2):
      if col+j<0 or col+j>=cur_lines[1].size:
        nbhd.append(NODATA)
      else:
        nbhd.append(cur_lines[i][col+j])

  dz_dx = (nbhd[2] + 2*nbhd[5] + nbhd[8] - (nbhd[0] + 2*nbhd[3] + nbhd[6])) \
                          / (8*cellsize)
  dz_dy = (nbhd[6] + 2*nbhd[7] + nbhd[8] - (nbhd[0] + 2*nbhd[1] + nbhd[2])) \
                            / (8*cellsize)

  slope = np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy)))

  return str(slope)

if __name__ == '__main__':
	main()
