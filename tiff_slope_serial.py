import numpy as np
from osgeo import gdal
import struct
import os

gdal.UseExceptions()

fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

def main():
	src_ds = gdal.Open("fredout.tif")
	prj = src_ds.GetProjection()
	
	i = src_ds.RasterCount
	srcband = src_ds.GetRasterBand(i)
	
	GeoT = src_ds.GetGeoTransform()
	NODATA = srcband.GetNoDataValue()
	xllcorner = GeoT[0]
	yllcorner = GeoT[3]
	cellsize = srcband.GetScale()
	ncols = srcband.YSize
	nrows = srcband.XSize
	BandType = gdal.GetDataTypeName(srcband.DataType)
	print NODATA, cellsize, ncols, nrows, BandType

	data = []

	for y in range(srcband.YSize):
		q = []
		for x in range(srcband.XSize):
			# get data from band and append it to list
			q.append(struct.unpack(fmttypes[BandType], srcband.ReadRaster(x,y,1,1, buf_type=srcband.DataType))[0])
		data.append(np.array(q, np.float64))
		
	#print data
	data = np.array(data, np.float64)
	#slope_data = calc_slope(data, cellsize, NODATA)
	#print len(slope_data), len(slope_data[0])

	#print slope_data
	array_to_raster(data, xllcorner, yllcorner, cellsize, ncols, nrows, prj, NODATA, GeoT)

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

def array_to_raster(array, xllcorner, yllcorner, cellsize, ncols, nrows, prj, NODATA, GeoT):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    dst_filename = 'output.tiff'


    # You need to get those values like you did.
    x_pixels = nrows  # number of pixels in x
    y_pixels = ncols  # number of pixels in y
    PIXEL_SIZE = cellsize  # size of the pixel...        
    x_min = xllcorner  
    y_max = yllcorner  # x_min & y_max are like the "top left" corner.
    wkt_projection = prj

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        int(x_pixels),
        int(y_pixels),
        1,
        gdal.GDT_Float32)

    #dataset.SetGeoTransform((
    #    x_min,    # 0
    #    PIXEL_SIZE,  # 1
    #    0,                      # 2
    #    y_max,    # 3
    #    0,                      # 4
    #    -PIXEL_SIZE))  
    dataset.SetGeoTransform(GeoT)
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.GetRasterBand(1).SetNoDataValue(NODATA)
    dataset.FlushCache()  # Write to disk.

if __name__ == '__main__':
	main()
