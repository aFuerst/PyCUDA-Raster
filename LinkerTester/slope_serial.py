import numpy as np
from os.path import abspath

def main(inputPath, outputPath):
	input_file = open(inputPath)
	params = input_file.readlines()
	input_file.close()
	ncols = params[0].split()[1]
	nrows = params[1].split()[1]
	xllcorner = params[2].split()[1]
	yllcorner = params[3].split()[1]
	cellsize = float(params[4].split()[1])
	NODATA = float(params[5].split()[1])

	# read data in as n by m list of numpy floats
	data = np.loadtxt(open(inputPath), skiprows=6)
  
	slope_data = calc_slope(data, cellsize, NODATA)
	#print slope_data
	np.savetxt(outputPath, slope_data, fmt='%5.2f')

def calc_slope(grid, cellsize, NODATA):
	slope_grid = np.zeros_like(grid)
	for row in range(len(grid)):
		for col in range((len(grid[0]))):
			slope_grid[row][col] = cell_slope(grid, row, col, cellsize, NODATA)

	return slope_grid

def cell_slope(grid, row, col, cellsize, NODATA):
	if grid[row][col] == NODATA:
		return NODATA
	
  #First, grab values for cells used in calculation
	nbhd = []
	for i in range(-1,2):
		for j in range(-1,2):
		#If out of bounds, log NODATA, these will be changed later.
			if row+i<=0 or row+i>=len(grid) or col+j<=0 or col+j>=len(grid[0]) or grid[row + i][col +j] == NODATA:
				nbhd.append(NODATA)
			else:
				nbhd.append(grid[row+i,col+j])

	dz_dx = (nbhd[2] + 2*nbhd[5] + nbhd[8] - (nbhd[0] + 2*nbhd[3] + nbhd[6])) \
				/ (8*cellsize)
	dz_dy = (nbhd[6] + 2*nbhd[7] + nbhd[8] - (nbhd[0] + 2*nbhd[1] + nbhd[2])) \
				/ (8*cellsize)

	slope = np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy)))

	return slope
	
if __name__ == '__main__':
	main()
