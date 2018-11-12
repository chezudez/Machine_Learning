#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:52:47 2018

@author: chezoudezue
"""
#The aim of this project is to create an Artificial Neural Network for a bank 
#the neural network can reduce customer churn by predicting if a customer will leave the bank or not

#data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv') 
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Lets make the ANN
#import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu',input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu'))
#add output layer
classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'sigmoid'))
#compiling the ANN
classifier.compile(optimizer ='adam',loss ='binary_crossentropy', metrics=['accuracy'] )

#fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch =100)


#Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""Assignment
Geo - France
Credit Score - 600
Gender - Male
age - 40
tenure - 3
balance - 60000
no of products - 2
Credit card - yes
estimated salary = 50k"""

new_case = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 50000]])))
new_case = (new_case > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)















