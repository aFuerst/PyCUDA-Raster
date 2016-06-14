#! /usr/bin/env python

import numpy as np

def main():

  input_file = open("aigrid.asc")
  params = input_file.readlines()
  input_file.close()
  ncols = params[0].split()[1]
  nrows = params[1].split()[1]
  xllcorner = params[2].split()[1]
  yllcorner = params[3].split()[1]
  cellsize = float(params[4].split()[1])
  NODATA = float(params[5].split()[1])

#raw_data = []
  #for line in params[6:]:
  #  raw_data.append(line.split())
  #data = np.asarray(raw_data)
  #data = data.astype(np.float)
  data = np.loadtxt(open("aigrid.asc"), skiprows=6)
  
  slope_data = calc_slope(data, cellsize, NODATA)
  #print slope_data
  #np.savetxt("output.txt", slope_data, fmt='%5.2f')

def calc_slope(grid, cellsize, NODATA):
  slope_grid = np.zeros_like(grid)
  for row in range(len(grid)):
    for col in range((len(grid[0]))):
      slope_grid[row][col] = cell_slope(grid, row, col, cellsize, NODATA)

  return slope_grid

def cell_slope(grid, row, col, cellsize, NODATA):
  #if row==len(grid)-1 or col==len(grid[0])-1 or row==0 or col==0:
    #TODO: Come back to this case later...
    #return NODATA

  #First, grab values for cells used in calculation
  nbhd = []
  for i in range(-1,2):
    for j in range(-1,2):
      #If out of bounds, log NODATA, these will be changed later.
      if row+i<=0 or row+i>=len(grid) or col+j<=0 or col+j>=len(grid[0]):
        nbhd.append(NODATA)
        continue
      nbhd.append(grid[row+i,col+j])

  if nbhd[4] == NODATA:
    return NODATA

  for i in range(len(nbhd)):
    if nbhd[i] == NODATA:
      nbhd[i] = nbhd[4]

  dz_dx = (nbhd[2] + 2*nbhd[5] + nbhd[8] - (nbhd[0] + 2*nbhd[3] + nbhd[6])) \
                / (8*cellsize)
  dz_dy = (nbhd[6] + 2*nbhd[7] + nbhd[8] - (nbhd[0] + 2*nbhd[1] + nbhd[2])) \
                / (8*cellsize)

  slope = np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy)))

  return slope

main()
