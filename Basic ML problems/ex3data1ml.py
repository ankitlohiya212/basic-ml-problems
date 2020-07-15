import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import scipy.io
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
Y = data['y']
Xtrain , Xtest,Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)
Xtrain , Xval , Ytrain , Yval = train_test_split(Xtrain,Ytrain,test_size=0.2,shuffle = True)

"""
from sklearn.svm import LinearSVC
clf = LinearSVC(multi_class = 'crammer_singer')
clf.fit(Xtrain, Ytrain.ravel())
Ypred = clf.predict(Xtest)

svmacc = accuracy_score(Ytest,Ypred)
print("Accuracy for SVMs=",svmacc)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs',multi_class='ovr').fit(Xtrain,Ytrain)
Ypred = lr.predict(Xtest)
lracc = accuracy_score(Ytest,Ypred)
print("Accuracy for Logistic Regression=",lracc)
"""
"""
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators=1000, max_depth=15)
clf1.fit(Xtrain, Ytrain.ravel())
Ypred= clf1.predict(Xtest)
rfacc = accuracy_score(Ytest,Ypred)
print("Accuracy for Random Forests",rfacc)


"""
Ytrain = np_utils.to_categorical(Ytrain)
Ytest = np_utils.to_categorical(Ytest)
Yval = np_utils.to_categorical(Yval)

model = Sequential()
model.add(Dense(400,input_shape=(400,),activation='relu',kernel_initializer='normal'))
model.add(Dense(11, activation = 'softmax',kernel_initializer='normal'))
model.compile(loss='categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
history = model.fit(Xtrain ,Ytrain,batch_size = 500 ,epochs=100,verbose=2, validation_data=(Xval,Yval))

acc = model.evaluate(x=Xtest,y=Ytest,batch_size=200,verbose=2)
Ypred = model.predict(X)
print('\n\n\nAccuracy = ', acc)

def display(i):
    img = X[i]
    plt.title('Example %d Label: %d Prediction : %d' % (i, Y[i], np.argmax(Ypred[i])))
    plt.imshow(img.reshape((20,20)),cmap=plt.cm.gray_r)
    plt.show()

