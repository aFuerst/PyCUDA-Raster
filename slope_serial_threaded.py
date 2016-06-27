import numpy as np
from multiprocessing import Process, Queue, Condition
from collections import deque
from time import sleep

"""
"Threaded" (actually processes) serial slope calculator for raster ascii files.
Uses producer-consumer paradigm to handle memory.

Charlie Kazer
"""

def main():

	input_file = open("aigrid.asc", 'r')
        output_file = open("output_slope.asc", 'w')

        #Create data buffer, conditions. Choose buffer size appropriate for memory size.
        #This process queue will handle locking automatically.
        data_buffer = Queue(maxsize=10000)

        #create processes
        load_proc = Process(target=load_func, args=(input_file, data_buffer))
        calc_proc = Process(target=calc_func, args=(output_file, data_buffer))

        load_proc.start()
        calc_proc.start()

        calc_proc.join()

#////////////////////////////////////////////////////////////////////////////////#

# load_func: Produces data from input file
def load_func(input_file, data_buffer):

  #Load file statistics data into buffer
  #////////////////////////////////////////////////////////////////////#
  ncols = input_file.readline().split()[1]
  nrows = input_file.readline().split()[1]
  xllcorner = input_file.readline().split()[1]
  yllcorner = input_file.readline().split()[1]
  cellsize = float(input_file.readline().split()[1])
  NODATA = float(input_file.readline().split()[1])

  data_buffer.put(int(ncols))
  data_buffer.put(int(nrows))
  data_buffer.put(float(xllcorner))
  data_buffer.put(float(yllcorner))
  data_buffer.put(float(cellsize))
  data_buffer.put(float(NODATA))
  #////////////////////////////////////////////////////////////////////#

  for line in input_file:
    data_buffer.put(np.fromstring(line, sep=' '))
  data_buffer.close()
  input_file.close()

#////////////////////////////////////////////////////////////////////////////////#

# calc_func: Consumes data from shared buffer. Calls calc_slope to handle actual calculations
# Writes results to output file.
def calc_func(output_file, data_buffer):

  #Get file statistics data from buffer, write back to output
  #////////////////////////////////////////////////////////////////////#
  numCols = data_buffer.get()
  numRows = data_buffer.get()
  xllcorner = data_buffer.get()
  yllcorner = data_buffer.get()
  cellsize = data_buffer.get()
  NODATA = data_buffer.get()

  #set up header
  header_str = ("ncols %d\n"
            "nrows %d\n"
            "xllcorner %f\n"
            "yllcorner %f\n"
            "cellsize %f\n"
            "NODATA_value %f\n"
            % (numCols, numRows, xllcorner, yllcorner, cellsize, NODATA)
            )

  output_file.write(header_str)
  #////////////////////////////////////////////////////////////////////#

  cur_lines = deque([], 3)
  count = 0
  cur_slope = []

  #First, insert a NODATA row
  nodata_row = np.zeros(numCols)
  nodata_row.fill(NODATA)
  cur_lines.append(nodata_row)
  #Read first two lines so that when we enter main while, cur_lines will
  #always contain 3 lines
  while(len(cur_lines) < 3):
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
    cur_lines.append(data_buffer.get())
    count += 1

    for i in range(cur_lines[1].size):
      cur_slope.append(calc_slope(cur_lines, i, cellsize, NODATA))
    output_file.write(' '.join(cur_slope))
    output_file.write('\n')
    cur_slope = []

  #Calculate slope for bottom line, add another NODATA row
  cur_lines.append(nodata_row)
  for i in range(cur_lines[1].size):
      cur_slope.append(calc_slope(cur_lines, i, cellsize, NODATA))
  output_file.write(' '.join(cur_slope))
  output_file.write('\n')
  cur_slope = []

  output_file.close()

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
