import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv('heart.csv' , encoding = 'latin-1')
data = np.matrix(data)
Y = data[:,13]
y = pd.DataFrame(data = Y)
Y = y.values
Y = np.transpose(Y)
Y = Y.ravel()

X = data[:,:13]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#X = preprocessing.normalize(X)
Xtrain, Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)


print("Using SVM....")
svm = SVC(kernel='rbf',gamma=5e-5,C=750)
svm.fit(Xtrain,Ytrain)
Ytrain_pred = svm.predict(Xtrain)
Ytest_pred = svm.predict(Xtest)
train_acc = accuracy_score(Ytrain , Ytrain_pred)
test_acc = accuracy_score(Ytest , Ytest_pred)
print("Training Accuracy for SVM is : " )
print(train_acc)
print("Test Accuracy for SVM is : " )
print(test_acc)

"""
param={'kernel':['rbf'],'C':[500,750,850,900,1000],'gamma':[0.000005,0.0000075,0.00001,0.000025,0.00005,0.000062]}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(),param,scoring='accuracy')
clf.fit(Xtrain,Ytrain)
print(clf.best_params_)
"""

print("Using NN.....")
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.models import Sequential
Ytrain = np_utils.to_categorical(Ytrain)
Ytest = np_utils.to_categorical(Ytest)
X = np.transpose(X)

model = Sequential()
model.add(Dense(13,input_shape=(13,),activation='relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss='categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
history = model.fit(Xtrain ,Ytrain,batch_size = 32 ,epochs=100,verbose=0, validation_data=(Xtest,Ytest))

