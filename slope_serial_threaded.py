import numpy as np
import threading
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
	NODATA = int(input_file.readline().split()[1])

        # NOTE: Don't skip any lines here, the file pointer has already advanced
        # past the header to the data.

        #Create lock and data buffer. Choose buffer size appropriate for memory size.
        lock = threading.Condition()
        data_buffer = deque([], 10000)

        output_file = open("output_slope.asc", 'w')
        
        #create threads
        load_thread = threading.Thread(target=load_func, args=(input_file, data_buffer, lock))
        calc_thread = threading.Thread(target=calc_func, args=(output_file, data_buffer, lock, int(nrows), int(ncols), float(cellsize), float(NODATA)))

        #Tell python that if we terminate our program prematurely, terminate threads.
        #load_thread.daemon = True
        #calc_thread.daemon = True

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
        load_thread.start()
        calc_thread.start()

        #This allows us to ctrl-c out. TODO: is there a more efficient way to do this?
        #while load_thread.isAlive() or calc_thread.isAlive():
        #  sleep(0.1)
        calc_thread.join()

        #clean-up
        input_file.close()
        output_file.close()


#////////////////////////////////////////////////////////////////////////////////#

# load_func: Produces data from input file
def load_func(input_file, data_buffer, lock):
  for line in input_file:
    lock.acquire()
    while (len(data_buffer) == data_buffer.maxlen):
      lock.wait()
    data_buffer.append(np.fromstring(line, sep=' '))
    lock.notify()
    lock.release()


#////////////////////////////////////////////////////////////////////////////////#

# calc_func: Consumes data from shared buffer. Calls calc_slope to handle actual calculations
# Writes results to output file.
def calc_func(output_file, data_buffer, lock, numRows, numCols, cellsize, NODATA):
  cur_lines = deque([], 3)
  count = 0
  cur_slope = []

  #Read first two lines so that when we enter main while, cur_lines will
  #always contain 3 lines
  while(len(cur_lines) < 2):
    lock.acquire()
    while (len(data_buffer) == 0):
      lock.wait()

    cur_lines.append(data_buffer.popleft())
    count += 1

    lock.notify()
    lock.release()

  #Calculate slope for top line
  cur_slope = calc_slope_edge(cur_lines, cellsize, NODATA, True)
  output_file.write(' '.join(cur_slope))
  output_file.write('\n')
  cur_slope = []

  #Main loop
  while (count < numRows):
    lock.acquire()
    while (len(data_buffer) == 0):
      lock.wait()

    cur_lines.append(data_buffer.popleft())
    count += 1

    lock.notify()
    lock.release()

    cur_slope = calc_slope(cur_lines, cellsize, NODATA)
    output_file.write(' '.join(cur_slope))
    output_file.write('\n')
    cur_slope = []

  #Calculate slope for bottom line
  cur_slope = calc_slope_edge(cur_lines, cellsize, NODATA, False)
  output_file.write(' '.join(cur_slope))
  output_file.write('\n')
  cur_slope = []



#////////////////////////////////////////////////////////////////////////////////#

# calc_slope: calculates the slope values for a row of data
# NOTE: We could change this to calculate the slope for a single cell.
# If we structured the program that way, we could plug in any raster calculation
# easily. However, I chose this so that we could calculate the edge cases more
# easily using calc_slope_edge below.
def calc_slope(cur_lines, cellsize, NODATA):
  slope_row = []
  for col in range(cur_lines[1].size):
    if cur_lines[1][col] == NODATA:
      slope_row.append(str(NODATA))
      continue

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
    slope_row.append(str(slope))

  return slope_row

#////////////////////////////////////////////////////////////////////////////////#

# calc_slope_edge: wrapper function which appends a NODATA row to the first
# and last rows of data. Then proceeds to call calc_slope.
def calc_slope_edge(cur_lines, cellsize, NODATA, top):
  new_lines = cur_lines
  if top:
    new_lines.appendleft(np.empty_like(new_lines).fill(NODATA))
    return calc_slope(new_lines, cellsize, NODATA)
  else:
    new_lines.append(np.empty_like(new_lines).fill(NODATA))
    return calc_slope(new_lines, cellsize, NODATA)

#////////////////////////////////////////////////////////////////////////////////#

if __name__ == '__main__':
	main()
