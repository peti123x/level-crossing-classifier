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

df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "length", "state", "wait-categ-none", "wait-categ-short", "wait-categ-medium", "wait-categ-long", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
#Generate or load based on this structure
import create_load_df as factory
df = factory.start(df)

print("Create classifiers... \n")

pd.set_option('display.expand_frame_repr', False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(df)
    print("")

save_df = df

iter_n = []
f1_macro = []
f1_weighted = []
for i in range(100, 2000, 25):
    third_class = i
    #There are only 1685 samples in this
    if i > 1680:
        third_class = 1680
    none_s = df.loc[df['wait-categ-none'] == 1].sample(n=i, random_state=1)
    short_s = df.loc[df['wait-categ-short'] == 1].sample(n=i, random_state=1)
    medium_s = df.loc[df['wait-categ-medium'] == 1].sample(n=third_class, random_state=1)
    long_s = df.loc[df['wait-categ-long'] == 1]
    
    df = pd.concat([none_s, short_s, medium_s, long_s])

    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    #input()

    x_labels = ['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    y_labels = ['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']

    train_x = train[x_labels]
    train_y = train[y_labels]

    x_labels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    train_x = train[x_labels]
    train_y = train[['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']]

    test_x = test[x_labels]
    test_y = test[y_labels]

    clf = RandomForestClassifier(criterion='entropy', max_features="auto", max_depth=None, min_samples_leaf=2, min_samples_split = 3, n_estimators=500)
    clf = clf.fit(train_x,train_y)
    pred_y = clf.predict(test_x)

    #print(accuracy_score(test_y, pred_y))
    print(classification_report(test_y, pred_y))
    asDict = classification_report(test_y, pred_y, output_dict=True)
    f1_macro_avg = asDict["macro avg"]["f1-score"]
    f1_weighted_avg = asDict["weighted avg"]["f1-score"]
    f1_macro.append(f1_macro_avg)
    f1_weighted.append(f1_weighted_avg)
    iter_n.append(i)
    #print(asDict)
    #print(f1_macro_avg)
    #print(f1_weighted_avg)

    df = save_df


    #NearestNeighbour
    #print("\n")
    #print("Nearest Neighbour")
    #clf = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")
    #clf = clf.fit(train_x, train_y)
    #pred_y = clf.predict(test_x)

    #print(accuracy_score(test_y, pred_y))
    #print(classification_report(test_y, pred_y))

plot.plot(iter_n, f1_macro, label="F1 macro")
plot.plot(iter_n, f1_weighted, label="F1 weighted ")
plot.ylabel("Performance")
plot.xlabel("Number of samples per class")
plot.axhline(y=np.max(f1_macro), label="F1 Macro max value", linestyle='--')
plot.axvline(x=1680, linestyle='-')
plot.legend()
plot.show()

input()
