# Lembo-REU-2016
Opensource GIS project leveraging NVIDIA CUDA and pyCuda

Students: Alex Fuerst, Charlie Kazer, Billy Hoffman

## Synopsis
A command line tool and plugin for [QGIS](http://www.qgis.org/) that leverages
a CUDA enabled GPU for raster calculations. Currently supports calculating slope,
aspect, and hillshade from an elevation file.

## Status
QGIS plugin is still in development, mosty untested and is unstable.
The command line scheduler supports calculating aspect, slope and 
hillshade on ESRI ascii and GEOTiff files.

##Dependencies

[Python 2.7.12](https://www.python.org/downloads/)

[CUDA 6.0 or higher](https://developer.nvidia.com/cuda-downloads/)

[Numpy](http://www.numpy.org/)

[Boost.Python](http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/index.html)
(Part of the Boost Libraries)
    
[Python GDAL](https://pypi.python.org/pypi/GDAL/)

[PyCUDA](https://mathema.tician.de/software/pycuda/)

This list may not include all of the dependencies the above programs require.

The gpustruct class included here was taken from another 
[git repo.](https://github.com/compmem/cutools)
Thanks to OSU!

##Installation

Simply download this git repository to your local machine and put them
wherever you want. If your CUDA installation folder isn't already set on your
PATH, the plugin may not work properly. If you want to use the QGIS plugin 
with a GUI prompt, copy the folder "CUDARaster" to the QGIS plugin folder.

On Linux the default location is 
```
/home/username/.qgis/python/plugins 
```
In Windows the default location is 
```
 C:\Users\username\\.qgis\python\plugins
```

##Use

This can be used via a command line interface or QGIS plugin. 

Plugin:
Has a simple GUI interface for chosing input and output files
and calculating slope, aspect and hillshade. The plugin supports
choosing between files on disk and layers already loaded into QGIS.
Currently using active layers isn't completely accurate and can not 
be guaranteed to calculate everything properly.

Command line:
```
python scheduler.py input_file output_1 func_1 output_2 func_2 ... output_n func_n
```
