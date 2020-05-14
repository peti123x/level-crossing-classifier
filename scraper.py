from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import date, datetime
import os
import time
from peewee import *
from time import gmtime, strftime
import keyboard
import schedule
import backup_db as backup

buffers = [{}, {}]

def clearBuffer():
    print("Buffer cleaned")
    for buffer in buffers:
       for key in list(buffer):
           del buffer[key]

def sendDB():
    backup.sendMail("trains.db")

schedule.every().day.at("01:00").do(sendDB)
schedule.every().day.at("02:33").do(clearBuffer)
#schedule.every(1).minutes.do(sendDB)

####Define DB stuff
db = SqliteDatabase('trains.db')
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

db.connect()
db.create_tables([Train])
####DB is defined

#Log will have the following format:
#Origin|Dest|Due|Delay|Barrier down|Barrier up|Barrier length|No trains passed

########
#Writes new entry to the DB
def writeRecord(og, dest, due, delay, est, canc, day):
    now = strftime("%Y/%m/%d", gmtime())
    train = Train(origin=og,
                  destination=dest,
                  due=due,
                  delay=delay,
                  est=est,
                  cancelled=canc,
                  day=day,
                  date=now)
    success = train.save()
    print(success)

#Gets certain entry from the database then updates it
def updateRecord(og, dest, due, delay, est, canc, day):
    for train in Train.select().where(Train.destination == dest, Train.origin ==og, Train.due == due, Train.day==day).order_by(Train.id.desc()).limit(1):
        train.delay = delay
        train.est = est
        train.cancelled = canc
        train.save()

#updateRecord("bla", "Sheffield", 0, 0, 0, 0)

#Return the day as a string
def nameOfDay():
    now = date.today()
    return now.strftime('%A').lower()

def getTrainList(url):
    try:
        conn = urlopen(url)
        page = conn.read()
        conn.close()
        soup = BeautifulSoup(page, 'html.parser')

        #soup = soup.prettify()
        table = soup.find("div", attrs={"class": "tbl-cont"})
        trains = True
        try:
            tablebody = table.find("tbody")
            return tablebody, trains
        except AttributeError:
            print("No trains coming.")
            trains = False
            return "", trains
    except:
        print("HTTP Error. Will try again ...")
        return "", False

def checkInBuffers(buffers, buffrstr, due, currUrl):
    isArriving = 1
    if "dep" in currUrl:
        isArriving = 0
    
    if due in buffers[isArriving]:
        if buffers[isArriving][due] != buffrstr:
            print("***")
            print("Buffer: " + str(buffers[isArriving][due]))
            print("String:" + buffrstr)
            buffer[due] = buffrstr
            #Existing string is not same as buffer entry generated now, so rewrite in db
            print("Updating train going from "+origin+ " to " + dest + ", due at " + due + ", delayed by " + str(delay) + " est to arrive at " + est)
            updateRecord(origin, dest, due, delay, est, cancelled, nameOfDay())
    #Doesnt exist yet, so add to buffer
    else:
        print("Key " + str(due) + " not in buffer")
        print(buffrstr)
        buffers[isArriving][due] = buffrstr
        #print(buffer[due])
        print("Adding train going from "+origin+ " to " + dest + ", due at " + due + ", delayed by " + str(delay) + " est to arrive at " + est)
        writeRecord(origin, dest, due, delay, est, cancelled, nameOfDay())
    return buffers

#url = "http://ojp.nationalrail.co.uk/service/ldbboard/dep/LCN"
#url = "https://ojp.nationalrail.co.uk/service/ldbboard/arr/LCN"
urls = ["https://ojp.nationalrail.co.uk/service/ldbboard/arr/LCN",
        "http://ojp.nationalrail.co.uk/service/ldbboard/dep/LCN"]
#buffers = [{}, {}]

while True:
    for url in urls:
        tablebody, trains = getTrainList(url)
        if trains:
            rows = tablebody.find_all("tr")
            #For each row
            for item in rows:
                #Due time
                due = item.find("td").text.strip()

                #Origin or destination depending on context
                #If looking at arriving link then all trains are going to Lincoln
                
                if "arr" in url:
                    dest = "LNC"
                    origin = item.select_one("td:nth-of-type(2)").text.replace(" ", "").replace("\n","")
                #If looking at departure, then all origin is from Lincoln
                elif "dep" in url:
                    dest = item.select_one("td:nth-of-type(2)").text.replace(" ", "").replace("\n","")
                    origin = "LNC"
                
                #Status, which contains forecasted arrival as well as minutes late or early
                status = item.select_one("td:nth-of-type(3)")
                #We will define delay in minutes
                delay = 0
                est = due
                cancelled = False
                if status.text == "On time":
                    delay = 0
                elif "Cancelled" in status.text:
                    delay = 9999
                    cancelled = True
                elif "late" in status.text:
                    est = status.text.split("  ")[0]
                    raw = status.find("span").text.split("\xa0") #\xa0 = br
                    delay = raw[0]
                elif "early" in status.text:
                    #TODO: Change estimated time to time - delay
                    raw = status.find("span").text.split("\xa0") #\xa0 = br
                    delay = -abs(int(raw[0]))
                #print("Train from "+origin+ " to " + dest + ", due at " + due + ", delayed by " + str(delay) + " est to arrive at " + est)

                #Create entry for buffer
                buffrstr = due+dest+origin+str(delay)+est+str(cancelled)
                buffers = checkInBuffers(buffers, buffrstr, due, url)
    #Iterate through dictionary by k,v
    for buffer in buffers:
        for key in list(buffer):
            #Now as H:M, split at : to get H and M as list items, conver hr to int
            now = strftime("%H:%M", gmtime()).split(":")
            now = int(now[0])
            #Get the train due time, which is the same as the now time
            trainTime = key.split(":")
            trainTime = int(trainTime[0])
            transformed = trainTime
            #If flips it over, then transform the time, this is the difficult case
            if trainTime + 2 >= 24:
                transformed = trainTime - 24 + 2
                #Compare accordingly
                if 24 - now < transformed:
                    del buffer[key]
                if (now >= 2) and (now <= 4) and (transformed >= 2) and (transformed <= 4) and (now >= transformed):
                    del buffer[key]
            #Otherwise do simple comparison
            else:
                transformed = trainTime + 2
                if now >= transformed:
                    del buffer[key]
    ts = strftime("%H:%M:%S", gmtime())
    print(str(ts) + ":" + str(buffers))
    schedule.run_pending()
    time.sleep(2)
