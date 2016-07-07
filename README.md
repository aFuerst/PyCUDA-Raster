# Lembo-REU-2016
Opensource GIS project leveraging NVIDIA CUDA and pyCuda
Students: Alex Fuerst, Charlie Kazer, Billy Hoffman


Status:
    QGIS plugin is still in development, mosty untested and is unstable.
    The command line scheduler supports calculating aspect, slope and 
    hillshade on ESRI ascii and GEOTiff files.

Dependencies:
    Python 2.7
    CUDA 6.0 or higher
        https://developer.nvidia.com/cuda-downloads
    Numpy
        http://www.numpy.org/
    Python GDAL
        https://pypi.python.org/pypi/GDAL/
    PyCUDA
        https://mathema.tician.de/software/pycuda/

Installation:

    Simply download this git repository to your local machine and put them
    wherever you want. If you want to use the QGIS plugin copy the folder
    "LinkerTester" to the QGIS plugin folder.

Use:

    This can be used via a command line interface or QGIS plugin. 
    
    Plugin:
    Has a simple GUI interface for chosing input and output files
     and only supports slope currently.

    Command line:
    python scheduler.py input_file output_1 func_1 output_2 func_2 ... output_n func_n
