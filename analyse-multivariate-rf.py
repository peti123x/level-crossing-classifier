#This script will be doing all the analysis.
#Takes in each day's data one by one, pads out the data so that it is disected into 1 minute
#intervals, attaches true/false labels to the barriers position, and trains a classifier
#based on this information. Then, given a time t it should be possible to predict the position.

#We do this for each day of the week to have a model for each.
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
import backup_db as backup

#Define struct
df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "length", "state", "wait-categ-none", "wait-categ-short", "wait-categ-medium", "wait-categ-long", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
#Generate or load based on this structure
import create_load_df as factory
df = factory.start(df)

print("Create classifiers... \n")

pd.set_option('display.expand_frame_repr', False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(df)
    print("")

#Downsampling
#none_s = df.loc[df['wait-categ-none'] == 1].sample(n=1000, random_state=1)
#short_s = df.loc[df['wait-categ-short'] == 1].sample(n=1000, random_state=1)
#medium_s = df.loc[df['wait-categ-medium'] == 1].sample(n=1000, random_state=1)
#long_s = df.loc[df['wait-categ-long'] == 1]
#save_df = df
#df = pd.concat([none_s, short_s, medium_s, long_s])

train, test = train_test_split(df, test_size=0.2, shuffle=True)
#input()

x_labels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
y_labels = ['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']

train_x = train[x_labels]
train_y = train[y_labels]

test_x = test[x_labels]
test_y = test[y_labels]

#Individual, chosen Random Forest model
#clf = RandomForestClassifier(criterion='entropy', max_features="auto", max_depth=None, min_samples_leaf=2, min_samples_split = 3, n_estimators=2000)
clf = RandomForestClassifier(criterion='entropy', max_features="auto", max_depth=None, min_samples_leaf=2, min_samples_split = 3, n_estimators=500)
clf = clf.fit(train_x,train_y)
pred_y = clf.predict(test_x)

#print(accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

input()

#Random forest CV
print("\n")
print("Random forest")
#Give eta
combinations = 5*7*4*6
trainingTimes = combinations*5
cores = 4
avgTrainTime = 55
eta = (trainingTimes/cores)*avgTrainTime
hrs,m,s = factory.convertFromSec(int(eta))
print("...")
print("CV expected to take " + str(int(hrs))+":"+str(int(m))+":"+str(int(s)))
print("...")
#Start cv
cvStart = timeit.default_timer()
clf = RandomForestClassifier(criterion='entropy', max_features="auto")
params = {
    "n_estimators": [500, 1000, 1500, 2000, 5000],
    "min_samples_leaf": [1, 2,3,4,8,10,14],
    "max_depth": [50, 80, 100, None],
    "min_samples_split": [2, 3, 4, 8, 10, 14]
}
#params = {
#    "n_estimators": [5,10],
#    "min_samples_leaf": [2,3],
#    "max_depth": [5, 10, None]
#}
cv = GridSearchCV(clf,params,cv=5, scoring='f1_micro', verbose=6, n_jobs=-1)
cv.fit(train_x,train_y)
factory.show(cv, "RF_CV")
print(cv)
cvEnd = timeit.default_timer()
runtime = cvEnd - cvStart
runhrs, runm, runs = factory.convertFromSec(runtime)
print("Execution finished in " +str(runtime) + "s.")
print(str(int(runhrs))+":"+str(int(runm))+":"+str(int(runs)))

input()
