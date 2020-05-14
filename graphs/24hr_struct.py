import numpy as np
import matplotlib.pyplot as plot
from datetime import date, datetime
import datetime
from time import gmtime, strftime
import glob

def readCSV(fname):   
    myFile = open(fname) 
    row =0 
    coords =[] 
    for line in myFile:
        #skip first line as it contains labels
        coords.append(line.rstrip().split(",")[:])
        row = row+1
        #coords[row] = line.rstrip().split(",")[:] 
    myFile.close()
    return coords

def getFilesFromDir(dirname):
    files = glob.glob(dirname + "*.csv")
    return files

def tsAsSeconds(ts):
    fmt = "%H:%M:%S"
    dt = datetime.datetime.strptime(ts, fmt)
    return dt.second + (dt.minute * 60) + (dt.hour * 60 *60)

files = getFilesFromDir("../datastream/tuesday/")
content = readCSV(files[3])
print(content)

seconds_in_day = 24*60*60
x = []
y = []
for entry in content:
    seconds = tsAsSeconds(entry[0])
    x.append(np.sin(2*np.pi*seconds/seconds_in_day))
    y.append(np.cos(2*np.pi*seconds/seconds_in_day))

fig = plot.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(x, y)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Sin against cos transform of barrier descension times")
ax.set_ylabel("cos time")
ax.set_xlabel("sin time")
plot.show()
