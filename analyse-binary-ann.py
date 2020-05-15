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

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K

#import backup_db as backup

#scoring functions for ANN
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

#Structure for the model
df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "state", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
import create_load_df as factory
df = factory.start(df, False)

#ds = transformData(df, content)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

print("Create classifiers... \n")

#Create train and test sets
train, test = train_test_split(df, test_size=0.2, shuffle=True)
xlabels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'state-lag','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

train_x = train[xlabels]
train_y = train['state']
test_x = test[xlabels]
test_y = test['state']

#Define properties of ANN
hiddenLayers = 1
neurons = 3
hidden_neurons = int(train_x.shape[0]/(3*(neurons+1)))
print(hidden_neurons)
print(train_y.value_counts())
ep = 100
opt = optimizers.SGD(lr=0.0005)
#opt = optimizers.Adam(learning_rate=0.005, beta_1 = 0.95, beta_2=0.995, amsgrad=False)

model = Sequential()
model.add(Dense(units=neurons, activation="relu", input_shape=(len(xlabels),)))

model.add(Dense(units=2*hidden_neurons, activation="relu", input_shape=(18632,)))

model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc',f1_m,precision_m, recall_m])
fit = model.fit(train_x, train_y, epochs=ep)
#Predict
pred = model.predict(test_x)
#Performance
loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, test_y, verbose=0)

print(model.summary())

#Plot performance
plot.plot(fit.history['acc'], label="Accuracy")
plot.plot(fit.history['f1_m'], label="F1 Mean")
plot.plot(fit.history['precision_m'], label="Precision")
plot.plot(fit.history['recall_m'], label="Recall")
plot.ylabel("Model accuracy")
plot.xlabel("Epoch")
plot.title("Model training accuracy over epochs")
plot.legend()
plot.show()

print("Loss: " + str(loss))
print("Accuracy: " + str(accuracy))
print("F1: " + str(f1_score))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

input()
