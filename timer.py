import scheduler
import time

iterations = 10

gargrid = "/home/afuerst1/Documents/gargrid.tif"
fredout = "/home/afuerst1/Documents/Lembo-REU-2016/fredout.tif"

"""
startAsc = time.time()

for i in range(iterations):
    scheduler.run("fredout.asc", ["output.asc"], ["hillshade"])
    
avgAscTime = (time.time() - startAsc) / iterations
"""

startTif = time.time()

for i in range(iterations):
    scheduler.run(gargrid, ["output.tif"], ["hillshade"])

avgTifTime = (time.time() - startTif) / iterations

"""
startTrips = time.time()

for i in range(iterations):
    scheduler.run("fredout.tif", ["hillshade.tif", "slope.tif", "aspect.tif"], ["hillshade", "slope", "aspect"])

avgTripsTime = (time.time() - startTrips) / iterations
"""

#print "Ascii times: %d mins, %d secs" % ((avgAscTime / 60), (avgAscTime % 60))
print "Tif times: %d mins, %d secs" % ((avgTifTime / 60), (avgTifTime % 60))
#print "3 Funcs tif times: %d mins, %d secs" % ((avgTripsTime / 60), (avgTripsTime % 60))

