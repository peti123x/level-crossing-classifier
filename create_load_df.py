from peewee import *
import numpy as np
import matplotlib.pyplot as plot
from datetime import date, datetime
import datetime
from time import gmtime, strftime
import pandas as pd
import seaborn as sns
from sklearn import tree
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import preprocessing
from progress.bar import Bar

import timeit

import backup_db as backup


def createFile(name):
    f = open(name+".log", "a")
    print("Created file "+name+".log")
    return f
def writeToFile(file, content):
    file.write(content+"\n")
def generateEntry(item):
    return str(item[0]) + ","+str(item[1])+","+str(item[2])

def convertFromSec(s):
    hrs = np.floor(s/3600)
    m = np.floor(((s/3600) - hrs)*60)
    s = np.floor(s -(hrs*3600)-(m*60))
    return hrs, m, s

def show(results, fname=None, send=False):
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']

    together = np.array([mean_score,std_score,params]).transpose()
    sorted_d = together[together[:,0].argsort()[::-1]]
    print(sorted_d)

    print(f'Best parameters are: {results.best_params_}')
    for item in sorted_d:
        print(f'{round(item[0],3)} accuracy on average.')
        break
    print("\n")

    i = 0
    print("Top candidates:")
    print("=============================================================")
    handler = createFile(fname)
    for item in sorted_d:
        if i < 20:
            print(f'{round(item[0],3)} + or -{round(item[1],3)} for the {item[2]}')
        if fname != None:
            writeToFile(handler, generateEntry(item))
        i = i + 1
    handler.close()
    if send:
        backup.sendMail(fname+".log", 0, "CV Results")

def report_average(*args):
    report_list = list()
    for report in args:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        data = np.delete(data, 0, 1).astype(float)
        avg_total = np.array([x for x in splited[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
        df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
        report_list.append(df)
    res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
    return res.rename(index={res.index[-1]: 'avg / total'})

def getFilesFromDir(dirname):
    files = glob.glob(dirname + "*.csv")
    return files

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

def addSecs(tm, secs):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    return datetime.datetime.strftime(fulldate, "%H:%M:%S")

def tsToStr(ts):
    return datetime.datetime.strftime(ts, "%H:%M:%S")

def tsAsSeconds(ts):
    fmt = "%H:%M:%S"
    dt = datetime.datetime.strptime(ts, fmt)
    return dt.second + (dt.minute * 60) + (dt.hour * 60 *60)

def transformData(df, content, day):
    
    fmt = "%H:%M:%S"
    #Look at opening time of first entry
    opening = content[0][0]
    prevClose = "00:00:00"
    secondsElapsed = 0
    for entry in content:
        #Convert opening time to datetime and count number of minutes
        opening = entry[0]
        dt = datetime.datetime.strptime(opening, fmt) - datetime.datetime.strptime(prevClose, fmt)
        #entries, outstanding = divmod(dt.seconds, 60)

        if prevClose == "00:00:00":
            prevClose = entry[1]
            #secondsElapsed += (entries*60) + outstanding
            secondsElapsed += dt.seconds
            continue
        ###
        time = dt.seconds
        df = df.append({"start": secondsElapsed, "close": secondsElapsed+time, "length": int(time), "state": False, day: True}, ignore_index=True)
        secondsElapsed += time

        #Now store the session info as True
        df = df.append({"start": secondsElapsed, "close": secondsElapsed+int(entry[2]), "length": int(entry[2]), "state": True, day: True}, ignore_index=True)
        secondsElapsed += int(entry[2])

        prevClose = entry[1]
    #Close to midnight
    dt = datetime.datetime.strptime("23:59:59", fmt) - datetime.datetime.strptime(content[-1][1], fmt)
    df = df.append({"start": secondsElapsed,"close": secondsElapsed+dt.seconds+1, "length": int(dt.seconds+1), "state": False, day: True}, ignore_index=True)
    secondsElapsed += dt.seconds+1
        
    return df


def createDF(df, autoSave, saveFileAs, multi=True):
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in weekdays:
        #print("Starting to process files...")
        files = getFilesFromDir("datastream/"+day+"/")
        #print("Processing " + day + ". " + str(len(files)) + " files in total.")

        bar = Bar('Processing ' + day, max=len(files))
        i = 1
        for file in files:
            #print("Processing " + str(i) + "/"+str(len(files))+"...")
            content = readCSV(file)
            df = transformData(df, content, day)
            #print(df.size)
            i = i + 1
            bar.next()
        #print(day + " finished.")
        bar.finish()
    
    
    #Then carry out sin/cos transform on start time
    seconds_in_day = 24*60*60

    #CHANGED np.sin(x) to np.sin(2*pi*x/seconds_in_day)
    df["start-sin"] = df['start'].transform(lambda x: np.sin(2*np.pi*x/seconds_in_day))
    df["start-cos"] = df['start'].transform(lambda x: np.cos(2*np.pi*x/seconds_in_day))
    df['close-sin'] = df['close'].transform(lambda x: np.sin(2*np.pi*x/seconds_in_day))
    df['close-cos'] = df['close'].transform(lambda x: np.cos(2*np.pi*x/seconds_in_day))
    #Fill all NaN values with False (this is for onehot encoding)
    df = df.fillna(False)
    #TODO: lag one-hot encoded state
    #Change the state variables
    #State = 0: This is still equivalent to False
    #State = 1: This is for range 0 < length <= 100
    #State = 2: This is for range 100 < length <= 300
    #State = 3: This is for range length > 300
    if multi:
        df.loc[(df['length'] > 300) & (df['state'] == True), "state"] = 3
        df.loc[(df['length'] <= 300) & (df['length'] > 100) & (df['state'] == True), "state"] = 2
        df.loc[(df['length'] <= 100) & (df['length'] > 0) & (df['state'] == True), "state"] = 1
        df.loc[df['state'] == False, "state"] = 0
        #
        df.loc[df['state'] == 3, "wait-categ-long"] = True
        df.loc[df['state'] == 2, "wait-categ-medium"] = True
        df.loc[df['state'] == 1, "wait-categ-short"] = True
        df.loc[df['state'] == 0, "wait-categ-none"] = True
        #normalise length
        min_max_scaler = preprocessing.MinMaxScaler()
        length_scaled = min_max_scaler.fit_transform(df[['length']])
        df['length'] = length_scaled
    #add lag features
    df = pd.concat([df, df['start-sin'].shift(1).rename("start-sin-lag"), df['start-cos'].shift(1).rename("start-cos-lag"), df['state'].shift(1).rename("state-lag")], axis=1)
    df = pd.concat([df, df['close-sin'].shift(1).rename("prev-close-sin"), df['close-cos'].shift(1).rename("prev-close-cos"), df['length'].shift(1).rename("prev-length")], axis=1)
    #state shift
    if multi:
        df = pd.concat([df, df["wait-categ-long"].shift(1).rename("categ-long-lag"), df["wait-categ-medium"].shift(1).rename("categ-medium-lag"),
                        df["wait-categ-short"].shift(1).rename("categ-short-lag"), df['wait-categ-none'].shift(1).rename("categ-none-lag")], axis=1)
    df = df.dropna()

    if autoSave == 0:
        saveName = saveFileAs
    else:
        saveName = input("Enter the name to save dataframe as")
    df.to_pickle(saveName + '.h5')
    print("Finished.")
    return df

def loadDF():
    name = input("Name of dataframe save?\n")
    print("Loading...")
    df = pd.read_pickle(name + '.h5')
    return df


def start(df, multi=True):
    autoSave = 0
    saveFileAs = "dataframe_" + datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")
    
    try:
        menuOption = int(input("Run (default) | Load dataframe (1)"))
    except ValueError:
        menuOption = 0

    if menuOption == 0:
        print("\nSave dataframe automatically?")
        try:
            autoSave = int(input("Yes (0) (default) | No (1)"))
        except ValueError:
            autoSave = 0
        if autoSave != 1:
            print("Enter the name of the file to save as or press enter for default name.")
            fname = input()
            if fname != '':
                saveFileAs = fname
                print("Going to save dataframe as " + saveFileAs + ".h5.\n")

    if menuOption != 1:
        df = createDF(df, autoSave, saveFileAs, multi)
    else:
        df = loadDF()
    return df




#Define dataframe
df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "length", "state", "wait-categ-none", "wait-categ-short", "wait-categ-medium", "wait-categ-long", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])


#For each weekday crawl through files, append to dataframe whilst one-hot encoding

