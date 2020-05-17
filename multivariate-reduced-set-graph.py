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

from progress.bar import Bar

import tensorflow.compat.v1 as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    #tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #fp = fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    #precision = tp/(tp+fp)
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

#Build the model functionally
def build_model():
    hiddenLayers = 1
    neurons = 500
    alpha = 7.5
    hidden_neurons = int(15000/(alpha*(neurons + 4)))
    #opt = optimizers.SGD(lr=0.05)
    opt = optimizers.Adam(learning_rate=0.005, amsgrad=False)

    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", input_shape=(12,)))

    model.add(Dense(units=2*hidden_neurons, activation="relu", input_shape=(18632,)))

    model.add(Dense(units=4, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy',f1_m,precision_m, recall_m])
    return model

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

steps = (2100 - 100)/10
bar = Bar('Processing ', max=steps)
counter = 1
#for i in range(100, 2100, 10):
for i in range(100, 2100, 10):
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

    #Define the epochs to reduce as sample size increases;
    #This gives each enough time to converge but not too much so it takes very long
    ep = int((steps / counter)*150)
    if ep > 5000:
        ep = 5000
    df = save_df
    counter = counter + 1

    model = build_model()
    fit = model.fit(train_x, train_y, epochs=ep)
    pred = model.predict(test_x)
    #convert predictions to 1d
    pred_y = np.argmax(pred, axis=1)
    #convert labels to 1d 
    test_y = np.argmax(test_y.to_numpy(), axis=1)

    #clf = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")
    #clf = clf.fit(train_x, train_y)
    #pred_y = clf.predict(test_x)

    #clf = RandomForestClassifier(criterion='entropy', max_features="auto", max_depth=None, min_samples_leaf=2, min_samples_split = 3, n_estimators=500)
    #clf = clf.fit(train_x,train_y)
    #pred_y = clf.predict(test_x)

    #print(accuracy_score(test_y, pred_y))
    #print(classification_report(test_y, pred_y))
    asDict = classification_report(test_y, pred_y, output_dict=True, zero_division=True)
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


    #print(accuracy_score(test_y, pred_y))
    #print(classification_report(test_y, pred_y))
    bar.next()

bar.finish()

print("\n")
plot.plot(iter_n, f1_macro, label="F1 macro")
plot.plot(iter_n, f1_weighted, label="F1 weighted ")
plot.ylabel("Performance")
plot.xlabel("Number of samples per class")
plot.axhline(y=np.max(f1_macro), label="F1 Macro max value", linestyle='--')
plot.axvline(x=1680, linestyle='--', label="$|c_3|$", color="black")
plot.legend()
plot.show()

input()
