#Creates class wrapper for simulation that is carried out
#around one given time. This class takes in an array of times
#[t1, t2, t3, t4] that represent either the entire day or part of the day.
#This is important because sometimes barriers are kept down
#if it isnt long before the next train comes or goes

'''start by laoding all data points for the giben date d, let set S
Process S and modify all estimated arrival by adding/subtracting delay
Delete any cancelled trains
Order by estimated arrival rather than 'due' (asc)

There are certain times that are measured:
- b_d, average time of arrival of a train after the barrier goes down
- b_u, average time elapsed between train passing and barrier ascending
- t_d, maximum time elapsed waiting for the next train

Then start processing the data points of S:
- Take data point d_0
- Then barrier goes down b_d before this train
- See what the next data point is
  - If it is within t_d seconds, then wait for it to 'pass', then discard
  - Go back and check if there is another one
- Wait b_u, then barrier goes up
- Next
'''

from peewee import *
import numpy as np
import matplotlib.pyplot as plot
from datetime import date, datetime
import datetime
from time import gmtime, strftime
import glob
import pandas as pd
import seaborn as sns

db = SqliteDatabase("trains.db")
class Train(Model):
    id = AutoField()
    origin = CharField()
    destination = CharField()
    due = TimeField()
    delay = IntegerField()
    est = TimeField()
    cancelled = BooleanField()
    day = CharField()
    date = DateField()

    class Meta:
        database = db

class Simulation:
    def __init__(self):
        self.delayChance = 0.3
        self.delayAmount = 10 #minutes, this is drawn from q-exp func
        self.breakdownRate = 0.01 #use ratio of cancelled trains from dataset
        self.delayChance = 0.7 #use delayed trains from dataset as ratio, should mean it gets more accurate for larger datasize

        self.averageWaitingTime = 35 #seconds, after barrier goes down
        self.minBarrUpWait = 17 #s
        self.maxBarrUpWait = 112 #s
        self.maximumTimeElapsed = 110 #seconds
        self.minimumTimeElapsed = 10 #seconds
        self.barrierUpTime = 8.75 #seconds

        self.passengerTrainPass = 8 #seconds


    def createFile(self, name, day):
        f = open("datastream/"+day+"/"+name+".csv", "a")
        print("Created file datastream/"+day+"/"+name+".csv")
        return f

    def writeToFile(self, file, content):
        file.write(content+"\n")
        
    def getData(self, date):
        self.data = []
        for train in Train.select().where(Train.date == date):
            if not train.cancelled:
                self.data.append(train)
        #print(self.data)
        if len(self.data) == 0:
            return False
        else:
            return True

    def cleanData(self):
        tformat = "%H:%M:%S"
        for train in self.data:
            if train.delay < 0:
                #Convert to time
                minutes = datetime.timedelta(minutes=abs(train.delay))
                #Shift back
                modded = datetime.datetime.combine(datetime.date(1,1,1),train.est) - minutes
                #Take time only
                modded_time = modded.time()
                #Modify
                train.est = modded_time
        ##Testing
        #for train in self.data:
        #    if train.delay < 0:
        #        print("Train due: ", str(train.due), ", delayed by " , str(train.delay)," ETA",str(train.est))

    def orderByETA(self):
        self.data.sort(key= lambda x: x.est)
        #for train in self.data:
        #    print(train.est)

    def generateEntry(self, start, end, numOfTrains):
        tformat = "%H:%M:%S"
        diff = (datetime.datetime.combine(date.min,end) - datetime.datetime.combine(date.min,start)).seconds
        entry = start.strftime(tformat)+","+end.strftime(tformat)+","+str(diff)+","+str(numOfTrains)
        return entry

    def nameOfDay(self, date):
        return datetime.datetime.strptime(date, '%Y/%m/%d').strftime("%A").lower()

    def genBarrierDownTime(self):
        alpha = 2.5
        beta = 6
        rand = np.random.beta(alpha, beta, 1)
        return int(((rand - 0)/(1 - 0)) * (self.maxBarrUpWait - self.minBarrUpWait) + self.minBarrUpWait)

    def process(self, date):
        print("process***")
        print(date)
        
        hasData = self.getData(date)
        if not hasData:
            return False
        hasData = True
        self.cleanData()
        self.orderByETA()
        #Stores what indices are processed, this is in case there are multiple
        #trains that are close enough to each other that are in the same session
        indexesProcessed = []

        #Create file
        diffFormat = datetime.datetime.strptime(date, '%Y/%m/%d').strftime("%Y-%m-%d")
        handler = self.createFile(diffFormat, self.nameOfDay(date))

        #For each element in the set
        for i in range(len(self.data)):
            #If current index has not been processed yet, otherwise ignore and continue
            if i not in indexesProcessed:
                #print("SESSION")
                #Define current object
                train = self.data[i]
                #Shift back to get when barrier is estimated to go down
                deltaSecond = datetime.timedelta(seconds=abs(self.genBarrierDownTime()))
                #print("Barrier down ", str(deltaSecond))
                barrierDown = (datetime.datetime.combine(datetime.date(5,5,5),train.est) - deltaSecond).time()
                #print("Barrier down at" , barrierDown)
                #Is next train within self.maximumTimeElapsed of this train?
                #+ random time accounting for trains that are close by 
                waitingTime = datetime.timedelta(seconds=abs(self.maximumTimeElapsed + self.genBarrierDownTime()))
                waits = True
                #If nextTrain's arrival time is before the time the train waits until, then its within the session
                #loops because it might be that multiple trains will follow
                extra = 0
                while waits:
                    #Convert waiting time to datetime object and add onto arrival time to get what time it'll "wait" until
                    try:
                        waitsUntil = (datetime.datetime.combine(datetime.date(1,1,1),self.data[i+extra].est) + waitingTime).time()
                        #print("This train arriving at ", self.data[i+extra].est, " waits until ", waitsUntil)
                        #If next trains arrival time is before the time that is waited until, then
                        #work out next index by adding 1 and go to next iter
                        nextTrain = self.data[i+extra+1]
                        if nextTrain.est <= waitsUntil:
                            #Barrier is still down
                            #print("Another train is arriving at ",nextTrain.est)
                            extra = extra + 1
                            indexesProcessed.append(i + extra)
                        else:
                            #If not within the time range, then the barrier is going up
                            indexesProcessed.append(i)
                            break
                        
                    except IndexError:
                        #If can't iterate to the next one, then there are no more trains in the set
                        #therefore end
                        #print("No next trains.")
                        break

                #TODO: Maybe these should be added individually after each train?
                trainPassTime = (extra+1)*self.passengerTrainPass
                overallShift = datetime.timedelta(seconds=(trainPassTime + self.barrierUpTime))
                lastTrain = self.data[i+extra]
                endTime = (datetime.datetime.combine(datetime.date(1,1,1),lastTrain.est)+overallShift).time()
                
                #Now write to file
                entry = self.generateEntry(barrierDown, endTime, extra+1)
                self.writeToFile(handler, entry)
                #print("Last train est ", lastTrain.est)
                #print("Barrier up at ", endTime) 
                #print("SESSION END")
                        
                #If yes then its within the same session

                #Otherwise barrier goes up and goes back down later
            else:
                continue
        handler.close()
        return hasData

    def processAll(self, date):
        hasDates = True
        currDate = date
        while hasDates:
            hasDate = self.process(currDate)
            if not hasDate:
                print("No more dates to process.")
                break
            nextDate = (datetime.datetime.strptime(currDate, '%Y/%m/%d') + datetime.timedelta(days=1)).strftime('%Y/%m/%d')
            currDate = nextDate
            print("Success.")
            print("******")
                        

    ##############################################
    def plotDelayAgainstDue(self, day):
        db.connect()
        #lastDate = Train.select().order_by(Train.date.desc()).get().date
        #firstDate = Train.select().order_by(Train.date.asc()).get().date
        #print(lastDate)
        #print(firstDate)
        print("Trains for " + day)
        y = []
        x = []
        dates = {}
        for train in Train.select().where(Train.day == day).order_by(Train.date.asc()):
            #print(train.date)
            if not train.cancelled:
                time = str(train.due).split(":")
                minute = float(time[1])/60
                overall = float(time[0])+minute
                if train.date in dates:
                    dates[train.date][1].append(train.delay)
                    #y.append(train.delay)
                    #x.append(float(overall))
                    dates[train.date][0].append(float(overall))
                #plot.scatter(float(overall), train.delay, label=train.date)
                else:
                    temp = []
                    temp.append([])
                    temp.append([])
                    dates[train.date] = temp
        for elem in dates.keys():
            plot.scatter(dates[elem][0], dates[elem][1], label=elem)
        plot.xlabel("Hours of the day")
        plot.ylabel("Delay")
        plot.xticks(np.arange(25))
        plot.legend()
        plot.show()
        print(dates)
    ##########

    def readCSV(self, fname):   
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

    def getAllFromDir(self, dirname):
        files = glob.glob(dirname + "*.csv")
        allData = []
        for file in files:
            allData.append(self.readCSV(file))
        return allData

    def padDs(self, data):
        padded = []
        for i in range(0,len(data)):
            #at start, pad from midnight to first opening
            if i == 0:
                entry = ["00:00:00",data[i][0],0,0]
            #then for all other data points, pad time between data[i][1] and data[i+1][0] with 0 interval length
            else:
                entry = [data[i-1][1], data[i][0], 0,0]
            padded.append(entry)
            padded.append(data[i])
        #print(padded)
        return padded

    def convertTime(self, t):
        time = str(t).split(":")
        if float(time[1]) == 0:
            minute = 0
        else:
            minute = float(time[1])/60
        if float(time[2]) == 0:
            second = 0
        else:
            second = float(time[2])/3600
        overall = float(time[0])+minute+second
        return overall

    #Plot as binary step graph
    def binaryGraph(self, day):
        fig = plot.figure(figsize=(17, 2))
        ax = fig.add_subplot(111)
        
        allData = self.getAllFromDir("datastream/"+day+"/")
        i = 1
        for ds in allData:
            padded = self.padDs(ds)
            x = []
            y = []
            for line in padded:
                x.append(line[0])
                y.append(int(line[2]))
                x.append(line[1])
                y.append(int(line[2]))

            ax.step(x,y, label=day+str(i))
            i = i + 1
            break
        
        plot.title(day)
        plot.xlabel("Time")
        plot.ylabel("Interval length (s)")
        plot.legend()

        plot.show()
    
    #With padded data
    def plotDay(self, day):
        allData = self.getAllFromDir("datastream/"+day+"/")
        print(allData)
        x = []
        y = []
        for ds in allData:
            padded = self.padDs(ds)
            for line in padded:
                #Append descending time as x
                time = str(line[0]).split(":")
                minute = float(time[1])/60
                second = float(time[2])/3600
                overall = float(time[0])+minute+second
                #x.append(overall)
                #print(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                x.append(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                #Append interval of staying down as y
                y.append(int(line[2]))
                
        df = pd.DataFrame({'barrier-descending':x, 'interval-length':y})
        df = df.sort_values(by=['barrier-descending'])
        df.plot('barrier-descending', 'interval-length', kind='line')
        plot.title(day)
        plot.xlabel("Barrier descending (hr)")
        plot.ylabel("Interval length (s)")
        plot.show()

    def plotDday(self, day):
        allData = simulation.getAllFromDir("datastream/"+day+"/")
        print(allData)
        x = []
        y = []
        for ds in allData:
            for line in ds:
                #Append descending time as x
                time = str(line[0]).split(":")
                minute = float(time[1])/60
                second = float(time[2])/3600
                overall = float(time[0])+minute+second
                #x.append(overall)
                #print(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                x.append(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                #Append interval of staying down as y
                y.append(int(line[2]))
        df = pd.DataFrame({'barrier-descending':x, 'interval-length':y})
        df = df.sort_values(by=['barrier-descending'])
        df.plot('barrier-descending', 'interval-length', kind='line')
        #sns.set(rc={'figure.figsize':(6,2)})
        #df['interval-length'].plot()
        plot.title(day)
        plot.xlabel("Barrier descending (hr)")
        plot.ylabel("Interval length (s)")
        plot.show()
        print(df)
        #plot.scatter(x,y)
        #plot.show()
        #plot.hist(y)
        #plot.show()

    def tsAsSeconds(self, ts):
        fmt = "%H:%M:%S"
        dt = datetime.datetime.strptime(ts, fmt)
        return dt.second + (dt.minute * 60) + (dt.hour * 60 *60)
    
    def plotIntervalAgainstDesc(self, days):
        print("Creating plot...")
        #print(allData)
        x = []
        y = []
        for day in days:
            allData = simulation.getAllFromDir("datastream/"+day+"/")
            for ds in allData:
                for line in ds:
                    #Append descending time as x
                    time = str(line[0]).split(":")
                    minute = float(time[1])/60
                    second = float(time[2])/3600
                    overall = float(time[0])+minute+second
                    #outliers
                    if int(line[2]) <= 1000:
                        x.append(overall)
                        #print(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                        #x.append(pd.to_datetime(line[0], format="%H:%M:%S").to_pydatetime().time())
                        #Append interval of staying down as y
                        y.append(int(line[2]))

        plot.scatter(x,y)
        plot.axhline(y=100, color='green')
        plot.axhline(y=300, color='yellow')

        plot.axhspan(0, 100, alpha=0.2, color='green', label="short wait")
        plot.axhspan(101, 300, alpha=0.2, color='yellow', label="medium wait")
        plot.axhspan(301, 800, alpha=0.2, color='tomato', label="long wait")
        
        plot.title("Interval length against barrier descension time")
        plot.xlabel("Barrier descending (hr)")
        plot.ylabel("Interval length (s)")
        
        plot.legend()
        plot.show()


simulation = Simulation()

#simulation.plotDelayAgainstDue("friday")
#simulation.binaryGraph("tuesday")

#Process trains
currDate = "2019/12/09"
simulation.processAll(currDate)
currDate = "2019/12/27"
simulation.processAll(currDate)
#coords = simulation.readCSV("datastream/friday/2019-12-13.csv")
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

simulation.plotIntervalAgainstDesc(days)
