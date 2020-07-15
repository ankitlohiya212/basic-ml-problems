from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data = loadmat('ex6data3.mat')
X= np.vstack((data['X'],data['Xval']))
Y = np.vstack((data['y'],data['yval']))
Y=Y.ravel()
Xtrain, Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.3,shuffle=True)

svm = SVC(kernel='rbf',gamma=256,C=1)
svm.fit(Xtrain,Ytrain.ravel())
Ytrain_pred = svm.predict(Xtrain)
Ytest_pred = svm.predict(Xtest)
Ypred = svm.predict(X)
train_acc = accuracy_score(Ytrain , Ytrain_pred)
test_acc = accuracy_score(Ytest , Ytest_pred)

print("Training Accuracy for SVM is : " )
print(train_acc)
print("Test Accuracy for SVM is : " )
print(test_acc)

plt.scatter(X[:,0],X[:,1],c=Y.ravel())
plt.show()

"""
param={'kernel':['rbf','linear'],'C':[0.1,1,5,10,100],'gamma':[64,128,256,512,1024]}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(),param,scoring='accuracy')
clf.fit(Xtrain,Ytrain)
print(clf.best_params_)
print(clf.best_score_)

"""
