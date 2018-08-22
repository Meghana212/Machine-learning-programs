import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#X_test1=X_test1[CreditScore = 600,Gender = 'Male',Age=40,Tenure=3,Balance=60000,NumOfProducts=2,HasCrCard=1,IsActiveMember=1,EstimatedSalary=50000]
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]#to prevent dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras and required modules
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#initializing the ANN
classifier = Sequential()
#adding the input layer and the first hidden layer. units = (11+1)/2. with dropout.
classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=1))
#adding second hidden layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu'))
classifier.add(Dropout(p=1))
#adding the output layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation='sigmoid'))
#compiling the ANN.Using logarithmic loss(could be binary or categorical)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred =(y_pred>0.5)

new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#evaluating the ann
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
#function to pass as argument 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu'))
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()

#improving the ann
#tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation='relu'))
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#used for wrapping
classifier = KerasClassifier(build_fn=build_classifier)

#create dictionary with all combinations of hyperparameters that we want to optimize.
#The values will be the values we want to try on the hyperparameters.(powers of 2 are sometimes taken as values)
parameters={'batch_size':[25,15],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator = classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters= grid_search.best_params_
best_accuracy=grid_search.best_score_











