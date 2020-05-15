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
xlabels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

train_x = train[xlabels]
train_y = train['state']
test_x = test[xlabels]
test_y = test['state']

print("Random forest")
#Give eta
combinations = 5*7*6*8
trainingTimes = combinations*5
cores = 4
avgTrainTime = 23
eta = (trainingTimes/cores)*avgTrainTime
hrs,m,s = factory.convertFromSec(int(eta))
print("...")
print("CV expected to take " + str(int(hrs))+":"+str(int(m))+":"+str(int(s)))
print("...")


#Start cv
cvStart = timeit.default_timer()
clf = RandomForestClassifier(criterion='entropy', max_features="auto")
params = {
    "n_estimators": [100, 200, 500, 1000, 2000],
    "min_samples_leaf": [1, 2,3,4,8,12,16],
    "max_depth": [20, 40, 60, 80, 100, None],
    "min_samples_split": [2, 3, 4,6,8,10,12,16]
}
#params = {
#    "n_estimators": [500],
#    "min_samples_leaf": [1, 2],
#    "max_depth": [None],
#    "min_samples_split": [2, 3]
#}
cv = GridSearchCV(clf,params,cv=5, scoring='f1', verbose=6, n_jobs=-1)
cv.fit(train_x,train_y)
#Output, set True as third parameter if you want results emailed, as this process is long
factory.show(cv, "binary_rf_cv")
cvEnd = timeit.default_timer()
runtime = cvEnd - cvStart
runhrs, runm, runs = factory.convertFromSec(runtime)
print("Execution finished in " +str(runtime) + "s.")
print(str(int(runhrs))+":"+str(int(runm))+":"+str(int(runs)))

input()
