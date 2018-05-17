"""
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  
On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, 
killing 1502 out of 2224 passengers and crew. 
This sensational tragedy shocked the international community and led to better 
safety regulations for ships.
One of the reasons that the shipwreck led to such loss of life was that there were not enough
lifeboats for the passengers and crew. Although there was some element of luck involved in 
surviving the sinking, some groups of people were more likely to survive than others, 
such as women, children, and the upper-class.
In this challenge, we ask you to complete the analysis of what sorts of people were likely to
survive. In particular, we ask you to apply the tools of machine learning to predict 
which passengers survived the tragedy.

@author: AMAN
"""
#Importing Lib
import numpy as np
import pandas as pd 

#Importing dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.dropna(subset=['Embarked'], how='any')
X_train= dataset.iloc[:,[2,4,5,6,7,11]].values 
y_train = dataset.iloc[:,1].values 

#Data Preprossing 
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X_train[:,2:3])
X_train[:,2:3]= imputer.transform(X_train[:,2:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X1 = LabelEncoder()
X_train[:,1] =  labelencoder_X1.fit_transform(X_train[:,1])
labelencoder_X2 = LabelEncoder()
X_train[:,5] =  labelencoder_X1.fit_transform(X_train[:,5])

onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()
onehotencoder5 = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder5.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

from sklearn.preprocessing import StandardScaler 
sc= StandardScaler()
X_train=sc.fit_transform(X_train)

import keras 
from keras.models import Sequential 
from keras.layers import Dense 

classifer = Sequential()
classifer.add(Dense(units=7,activation='relu', kernel_initializer='uniform', input_dim=13 ))
classifer.add(Dense(units=7,activation='relu', kernel_initializer='uniform' ))
classifer.add(Dense(units=7,activation='relu', kernel_initializer='uniform' ))
classifer.add(Dense(units=7,activation='relu', kernel_initializer='uniform' ))
classifer.add(Dense(units=1,activation='sigmoid', kernel_initializer='uniform'))
classifer.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifer.fit(X_train,y_train,batch_size=10,epochs=100)

dataset2 = pd.read_csv('test.csv')
X_test= dataset2.iloc[:,[1,3,4,5,6,10]].values 
dataset3 = pd.read_csv('gender_submission.csv')
y_test = dataset3.iloc[:,1].values 

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X_test[:,2:3])
X_test[:,2:3]= imputer.transform(X_test[:,2:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X3 = LabelEncoder()
X_test[:,1] =  labelencoder_X3.fit_transform(X_test[:,1])
labelencoder_X4 = LabelEncoder()
X_test[:,5] =  labelencoder_X4.fit_transform(X_test[:,5])

onehotencoder2 = OneHotEncoder(categorical_features = [0])
X_test = onehotencoder2.fit_transform(X_test).toarray()
onehotencoder3 = OneHotEncoder(categorical_features = [5])
X_test = onehotencoder3.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

from sklearn.preprocessing import StandardScaler 
sc= StandardScaler()
X_test=sc.fit_transform(X_test)

y_pred = classifer.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)










