#This script will be doing all the analysis.
#Takes in each day's data one by one, pads out the data so that it is disected into 1 minute
#intervals, attaches true/false labels to the barriers position, and trains a classifier
#based on this information. Then, given a time t it should be possible to predict the position.

#We do this for each day of the week to have a model for each.
import tensorflow.compat.v1 as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ctypes

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

#Macro, much lower
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

#Save dataframe
def build_model():
    hiddenLayers = 1
    neurons = 500
    alpha = 7.5
    hidden_neurons = int(15000/(alpha*(neurons + 4)))
    #opt = optimizers.SGD(lr=0.05)
    opt = optimizers.Adam(learning_rate=0.00005, amsgrad=False)

    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", input_shape=(12,)))

    model.add(Dense(units=2*hidden_neurons, activation="relu", input_shape=(18632,)))

    model.add(Dense(units=4, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy',f1_m,precision_m, recall_m])
    return model


df= pd.DataFrame(columns=["start-sin", "start-cos","start", "close-sin", "close-cos", "close", "length", "state", "wait-categ-none", "wait-categ-short", "wait-categ-medium", "wait-categ-long", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])


#import backup_db as backup
import create_load_df as factory

df = factory.start(df)

print("Create classifiers... \n")

pd.set_option('display.expand_frame_repr', False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
    print("")

#content = readCSV(files[0])
#ds = transformData(df, content)

#train, test = train_test_split(df, test_size=0.1, shuffle=True)

#train_x = train[['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'state-lag', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']]
#train_y = train[['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']]
#test_x = test[['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'state-lag', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']]
#test_y = test[['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']]
#test_y = test_y.replace(False, 0).replace(True,1)
#train_y = train_y.replace(False, 0).replace(True,1)


df = shuffle(df)

#x = df[['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'state-lag', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']]
#y = df[['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']]
#enforce, this is gone wrong somewhere
#y = y.replace(False, 0)
#y = y.replace(True, 1)


#old
#x_labels = ['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

#mcm_ann_t3
x_labels = ['start-sin', 'start-cos', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
#mcm_ann_t4
#x_labels = ['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
#
#y_labels = ['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']
y_labels = ['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']

#Take some entries out to see how a balanced dataset performs
#none_s = df.loc[df['wait-categ-none'] == 1].sample(n=1900, random_state=1)
#short_s = df.loc[df['wait-categ-short'] == 1].sample(n=1900, random_state=1)
#medium_s = df.loc[df['wait-categ-medium'] == 1]
#long_s = df.loc[df['wait-categ-long'] == 1]
#save_df = df
#df = pd.concat([none_s, short_s, medium_s, long_s])


print(len(df.loc[df['wait-categ-none'] == 1]))
print(len(df.loc[df['wait-categ-short'] == 1]))
print(len(df.loc[df['wait-categ-medium'] == 1]))
print(len(df.loc[df['wait-categ-long'] == 1]))

print("\n")
ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )
#input()

model = build_model()
train, test = train_test_split(df, test_size=0.2, shuffle=True)
train_x = train[x_labels]
train_y = train[y_labels]
test_x = test[x_labels]
test_y = test[y_labels]
fit = model.fit(train_x, train_y, epochs=500)
pred = model.predict(test_x)
#convert predictions to 1d
predicted_classes = np.argmax(pred, axis=1)
#convert labels to 1d 
un_onehot = np.argmax(test_y.to_numpy(), axis=1)
#score per class!
print(classification_report(un_onehot, predicted_classes))

ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True )

plot.plot(fit.history['f1_m'], label="F1 Mean")
plot.ylabel("Model F1 mean")
plot.xlabel("Epoch")
plot.title("Model training f1 mean over epochs")
plot.legend()
plot.show()
#input()

ep = 500
n_fold = 5
df_split = np.array_split(df, n_fold)
test_part = 0
acc = []
f1 = []
prec = []
recalls = []

acc_history = []
f1_history = []
prec_history = []
recall_history = []
while test_part < n_fold:
    model = build_model()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    print("CV Fold " , test_part, "/",n_fold)
    
    for i in range(n_fold):
        #on first iter that isnt a test part then set the train set to this 
        if len(train_x) == 0 and not i == test_part:
            train_x = df_split[i][x_labels]
            train_y = df_split[i][y_labels]
            #terminate immediately
            continue
        #if current is not a test partition then concat with previous version
        if not i == test_part:
            train_x = pd.concat([train_x, df_split[i][x_labels]], axis=0)
            train_y = pd.concat([train_y, df_split[i][y_labels]], axis=0)
            #train_x.append(df_split[i][['start-sin', 'start-cos', 'start-sin-lag', 'start-cos-lag', 'prev-close-sin', 'prev-close-cos', 'prev-length', 'state-lag', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']])
            #train_y.append(df_split[i][['wait-categ-none', 'wait-categ-short', 'wait-categ-medium', 'wait-categ-long']])
        #set this to test partition
        else:
            test_x = df_split[i][x_labels]
            test_y = df_split[i][y_labels]
    #enforce
    train_y = train_y.replace(False, 0)
    train_y = train_y.replace(True, 1)
    test_y = test_y.replace(False, 0)
    test_y = test_y.replace(True, 1)
    #fit
    fit = model.fit(train_x, train_y, epochs=ep, verbose=1)
    pred = model.predict(test_x)
    
    #score
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_x, test_y, verbose=0)
    #save
    acc_history.append(fit.history['categorical_accuracy'])
    f1_history.append(fit.history['f1_m'])
    prec_history.append(fit.history['precision_m'])
    recall_history.append(fit.history['recall_m'])
    acc.append(accuracy)
    f1.append(f1_score)
    prec.append(precision)
    recalls.append(recall)  
    test_part += 1
print("CV finished.\n")

print("Mean Accuracy")
print(sum(acc)/len(acc))
print("Mean F1 score")
print(sum(f1)/len(f1))
print("Mean Precision")
print(sum(prec)/len(prec))
print("Mean Recall rate")
print(sum(recalls)/len(recalls))

print("\n First model properties")
print(classification_report(un_onehot, predicted_classes))


fig, ((ax1, ax2), (ax3, ax4)) = plot.subplots(2,2, figsize=(12,6))
for i in range(len(acc_history)):
    ax1.plot(acc_history[i], label="Fold "+str(i+1))
    ax2.plot(f1_history[i], label="Fold " + str(i+1))
    ax3.plot(prec_history[i], label="Fold "+str(i+1))
    ax4.plot(recall_history[i], label="Fold " +str(i+1))
    #plot.plot(recall_history[i], label="Recall Fold "+str(i+1))
    #plot.plot(prec_history[i], label="Precision Fold"+str(i+1), color=plot.gca().lines[-1].get_color())
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.set_title("Categorical accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Performance")

ax2.set_title("F1-Micro average")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Performance")

ax3.set_title("Precision")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Performance")

ax4.set_title("Recall")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Performance")

plot.show()

input()

#Built in CV that does not work with other metrics
#classifier = KerasClassifier(build_fn=build_model, batch_size=1000, epochs=ep)
#accuracies = cross_val_score(estimator=classifier, X=x, y=y, cv=10, verbose=5)

#print(accuracies)
#print("Accuracy mean: " + str(accuracies.mean()))
#print("Accuracy std: " + str(accuracies.std()))

#For individual model
#plot.plot(fit.history['acc'], label="Accuracy")
#plot.plot(fit.history['f1_m'], label="F1 Mean")
#plot.plot(fit.history['precision_m'], label="Precision")
#plot.plot(fit.history['recall_m'], label="Recall")
#plot.ylabel("Model accuracy")
#plot.xlabel("Epoch")
#plot.title("Model training accuracy over epochs")
#plot.legend()
#plot.show()
#print("Loss: " + str(loss))
#print("Accuracy: " + str(accuracy))
#print("F1: " + str(f1_score))
#print("Precision: " + str(precision))
#print("Recall: " + str(recall))

#input()

