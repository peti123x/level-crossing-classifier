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

#Sample for smaller data set (this can be commented out
none_s = df.loc[df['wait-categ-none'] == 1].sample(n=1000, random_state=1)
short_s = df.loc[df['wait-categ-short'] == 1].sample(n=1000, random_state=1)
medium_s = df.loc[df['wait-categ-medium'] == 1].sample(n=1000, random_state=1)
long_s = df.loc[df['wait-categ-long'] == 1]
save_df = df
df = pd.concat([none_s, short_s, medium_s, long_s])

#Set up train and test
train, test = train_test_split(df, test_size=0.2, shuffle=True)

x_labels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
y_labels = ['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']

train_x = train[x_labels]
train_y = train[y_labels]

test_x = test[x_labels]
test_y = test[y_labels]


#NearestNeighbour individual, chosen model
print("\n")
print("Nearest Neighbour")
clf = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)

#print(accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

input()

#NearestNeighbour CV
print("\n")
print("Nearest Neighbour")
#Give ETA
combinations = 2*9*4*2
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
params = {
    "weights": ["uniform", "distance"],
    "n_neighbors": [2, 3, 5, 7, 9, 12, 15, 18, 21],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2]
}
#params = {
#    "weights": ["uniform", "distance"],
#    "n_neighbors": [2,3, 5, 7, 9]
#}
clf = KNeighborsClassifier()
cv = GridSearchCV(clf,params,cv=5, scoring='f1_micro', verbose=5, n_jobs=-1)
cv.fit(train_x,train_y)
factory.show(cv, "NN_CV")
cvEnd = timeit.default_timer()
runtime = cvEnd - cvStart
runhrs, runm, runs = factory.convertFromSec(runtime)
print("Execution finished in " +str(runtime) + "s.")
print(str(int(runhrs))+":"+str(int(runm))+":"+str(int(runs)))

input()
