import numpy as np

def main():

	#Get key values from input file
	input_file = open("aigrid.asc", 'r')
	ncols      = int(input_file.readline().split()[1])
	nrows      = int(input_file.readline().split()[1])
	xllcorner  = input_file.readline().split()[1]
	yllcorner  = input_file.readline().split()[1]
	cellsize   = float(input_file.readline().split()[1])
	NODATA     = int(input_file.readline().split()[1])
	
	#load data from input file
	data = np.loadtxt(input_file)
	input_file.close()

	#calculate slope and/or aspect and/or hillShade
#	slope_output     = return_slope(data, cellsize, NODATA)
#	aspect_output    = return_aspect(data, cellsize, NODATA)	
	hillShade_output = calc_hillShade(data, cellsize, NODATA)

	#store header info into string variable
	header_str = ("ncols %s\n"
                  "nrows %s\n"
                  "xllcorner %s\n"
                  "yllcorner %s\n"
                  "cellsize %f\n"
                  "NODATA_value %d"
                  % (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA)
                 )

	#output slope and/or aspect and/or hillShade info to output file
#	np.savetxt("aiSlopeOutput.asc", slope_output, fmt='%5.2f', header=header_str, comments='')
#	np.savetxt("aiAspectOutput.asc", aspect_output, fmt='%5.2f', header=header_str, comments='')
	np.savetxt("aiHillShade.asc", hillShade_output, fmt='%5.2f', header=header_str, comments='')

#calculate the hillShade from the data
def calc_hillShade(data, cellsize, NODATA):
	hillShade_data = np.zeros_like(data)
	for row in range(len(data)):
		for col in range((len(data[0]))):
			hillShade_data[row][col] = cell_shade(data, row, col, cellsize, NODATA)

	return hillShade_data

#calculate the hillshade for an indivdual cell
def cell_shade(data, row, col, cellsize, NODATA):
	if data[row][col] == NODATA:
		return NODATA

	#calculate zenith
	altitude = 45
	zenith_deg = (90 - altitude)
	zenith_rad = (zenith_deg * (np.pi / 180.0))
	
	#calculate azimuth	
	azimuth = 315
	azimuth_math = ((360.0 - azimuth) + 90)
	if azimuth_math >= 360.0:
		azimuth_math = (azimuth_math - 360.0)
	azimuth_rad = (azimuth_math * (np.pi / 180.0))

	#caluclate slope
	slope = calc_slope(data, row, col, cellsize, NODATA)

	#calculate aspect
	aspect = calc_aspect(data, row, col, cellsize, NODATA)

	#calculate hillShade
	hillShade = (255.0 * ((np.cos(zenith_rad) * np.cos(slope)) + (np.sin(zenith_rad) * np.sin(slope) \
				* np.cos(azimuth_rad - aspect))))
	

	return hillShade

#caluclate the slope for all the data
def return_slope(data, cellsize, NODATA):
	slope_data = np.zeros_like(data)
	for row in range(len(data)):
		for col in range((len(data[0]))):
			slope_data[row][col] = calc_slope(data, row, col, cellsize, NODATA)

	return slope_data

#calculate the slope for an indivdual cell
def calc_slope(data, row, col, cellsize, NODATA):
	if data[row][col] == NODATA:
		return NODATA

	nbhd = []
	for i in range(-1,2):
		for j in range(-1,2):
			if row+i<=0 or row+i>=len(data) or col+j>=len(data[0]) or data[row+i][col+j] == NODATA:
				nbhd.append(NODATA)
			else:
				nbhd.append(data[row+i,col+j])
		
	dz_dx = (nbhd[2] + 2*nbhd[5] + nbhd[8] - (nbhd[0] + 2*nbhd[3] + nbhd[6])) \
				/ (8*cellsize)
	dz_dy = (nbhd[6] + 2*nbhd[7] + nbhd[8] - (nbhd[0] + 2*nbhd[1] + nbhd[2])) \
				/ (8*cellsize)

	slope = np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy)))

	return slope

#caluclate the aspect for all the data
def return_aspect(data, cellsize, NODATA):
	aspect_data = np.zeros_like(data)
	for row in range(len(data)):
		for col in range((len(data[0]))):
			aspect_data[row][col] = calc_aspect(data, row, col, cellsize, NODATA)

	return aspect_data

#caluclate the aspect for an individual cell
def calc_aspect(data, row, col, cellsize, NODATA):
	if data[row][col] == NODATA:
		return NODATA 
	
	nbhd = []
	for i in range(-1,2):
		for j in range(-1,2):
			if row+i<=0 or row+i>=len(data) or col+j>=len(data[0]) or data[row+i][col+j] == NODATA:
				nbhd.append(NODATA)
			else:
				nbhd.append(data[row+i,col+j])

	dz_dx = (nbhd[2] + 2*nbhd[5] + nbhd[8] - (nbhd[0] + 2*nbhd[3] + nbhd[6])) \
				/ (8*cellsize)
	dz_dy = (nbhd[6] + 2*nbhd[7] + nbhd[8] - (nbhd[0] + 2*nbhd[1] + nbhd[2])) \
				/ (8*cellsize)
		
	aspect = (57.29578 * (np.arctan2(dz_dy, -(dz_dx))))	

	if aspect < 0:
		aspect = 90.0 - aspect
	elif aspect > 90.0:
		aspect = 360.0 - aspect + 90.0
	else:
		aspect = 90.0 - aspect

	aspect = (aspect * (np.pi / 180.0))

	'''	
	if dz_dx != 0:
		aspect = np.arctan2(dz_dy, -(dz_dx))
		if aspect < 0:
			aspect = ((2 * np.pi) + aspect)
	if dz_dx == 0:
		if dz_dy > 0:
			aspect = (np.pi / 2)
		elif dz_dy < 0:
			aspect = ((2 * np.pi) - (np.pi / 2))
		else:
			aspect = aspect
	'''
	
	return aspect


if __name__ == '__main__':
	main()
