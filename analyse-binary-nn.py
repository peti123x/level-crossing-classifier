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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import timeit

#import backup_db as backup

df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "state", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
import create_load_df as factory
df = factory.start(df, False)

#ds = transformData(df, content)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

print("Create classifiers... \n")

train, test = train_test_split(df, test_size=0.2, shuffle=True)
xlabels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

train_x = train[xlabels]
train_y = train['state']
test_x = test[xlabels]
test_y = test['state']

print("Nearest Neighbour")
#Calc estimated time
combinations = 2*14
trainingTimes = combinations*5
cores = 4
avgTrainTime = 0.9
eta = (trainingTimes/cores)*avgTrainTime
hrs,m,s = factory.convertFromSec(int(eta))


print("...")
print("CV expected to take " + str(int(hrs))+":"+str(int(m))+":"+str(int(s)))
print("...")
#Start CV
cvStart = timeit.default_timer()
#params to cross validate
params = {
    "weights": ["uniform", "distance"],
    "n_neighbors": [2, 3, 5, 7, 9, 12, 15, 18, 21, 23, 25, 27, 29, 31]
}
#Init CV
clf = KNeighborsClassifier()
cv = GridSearchCV(clf,params,cv=5, scoring='f1', verbose=5, n_jobs=-1)
cv.fit(train_x,train_y)
#output results
factory.show(cv, "binary_nn_cv")
cvEnd = timeit.default_timer()
runtime = cvEnd - cvStart
runhrs, runm, runs = factory.convertFromSec(runtime)
print("Execution finished in " +str(runtime) + "s.")
print(str(int(runhrs))+":"+str(int(runm))+":"+str(int(runs)))


input()
