import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import pandas as pd
import os
from tqdm import tqdm
from sklearn import metrics as ms
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
import sys

#CNN

data = os.listdir('I:/python/Dataset/')
classes={'0':0,'1':1, '2':2,'3':3, '4':4 , '5':5 , '6':6 , '7':7 , '8':8 , '9':9}
X=[]
Y=[]
Data = []
for cls in classes:
    dt ='I:/python/Dataset/'+cls
    for j in os.listdir(dt):
        img = cv2.imread(dt+'/'+j)
        res = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(i)
        # plt.show()
        X.append(res)
        Y.append(classes[cls])

X = np.array(X, dtype=object).astype(np.float32)
Y = np.array(Y)
"""Important note"""
"""when uncomment svm to run it the Y = np.expand_dims(Y, -1) line should be commented with the 
 rest of CNN model because it's only specific for CNN """
Y = np.expand_dims(Y, -1)

xtrain, xtest, ytrain, ytest = train_test_split(X ,Y, random_state=0, test_size=0.2, shuffle=True)

#Normalization
arr_mean = np.mean(xtrain, axis=(0))
# print(arr_mean.shape)
xtrain = abs(xtrain - arr_mean)/255.0
xtest = abs(xtest - arr_mean)/255.0
# print(xtrain.shape)
# print(ytrain.shape)
# """## Build the models"""
from sklearn.model_selection import KFold
num_folds = 3
fold_no = 1
acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=num_folds, shuffle=True)
for train, test in kfold.split(xtrain, ytrain):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(xtrain.shape[1:])))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    history = model.fit(xtrain, ytrain, epochs=10,
                        validation_data=(xtest, ytest))

    scores = model.evaluate(xtrain[test], ytrain[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    predictions1 = model.predict(xtest).argmax(axis=1)
    print("Accuracy:", ms.accuracy_score(ytest, predictions1))
    precision = precision_score(ytest, predictions1, average="weighted")
    print('Precision: %.3f' % precision)
    recall = recall_score(ytest, predictions1, average='weighted')
    print('recall: %.3f' % recall)
    score = f1_score(ytest, predictions1, average='macro')
    print('F-Measure: %.3f' % score)

###############################################################
# SVM
# x_train = np.reshape(xtrain, (len(xtrain), 30000))
# x_test = np.reshape(xtest, (len(xtest), 30000))
# X2 = np.reshape(X, (2062, 30000))
# svc = SVC(kernel='linear',gamma='auto')
# #svc = SVC(kernel='sigmoid',gamma='auto')
# #svc = SVC(kernel='rbf',gamma='auto')
# #svc = SVC(kernel='poly',gamma='auto')
# svc.fit(x_train, ytrain)
#
# y_predict = svc.predict(x_test)
#
# k_folds = KFold(n_splits=5)
# scores = cross_val_score(svc, X2, Y, cv=k_folds)
#
# print("Accuracy : ", accuracy_score(ytest,y_predict))
# precision = precision_score(ytest,y_predict,average = "weighted")
# print('Precision: %.3f' % precision)
# recall = recall_score(ytest,y_predict, average='weighted')
# print('recall: %.3f' % recall)
# score = f1_score(ytest,y_predict, average='macro')
# print('F-Measure: %.3f' % score)
# print("Cross Validation Scores: ", scores)

