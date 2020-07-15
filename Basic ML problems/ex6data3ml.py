import scipy.io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = scipy.io.loadmat('ex8_movies.mat')
X= data['R']
Y=data['Y']
#Y=Y.ravel()
Xtrain, Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)

svm = SVC(kernel='rbf',gamma=0.005,C=20)
svm.fit(Xtrain,Ytrain)
Ytrain_pred = svm.predict(Xtrain)
Ytest_pred = svm.predict(Xtest)
train_acc = accuracy_score(Ytrain , Ytrain_pred)
test_acc = accuracy_score(Ytest , Ytest_pred)
print("Training Accuracy for SVM is : " )
print(train_acc)
print("Test Accuracy for SVM is : " )
print(test_acc)

plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()



"""
param={'kernel':['rbf'],'C':[10,20,50],'gamma':[0.001,0.0025,0.005,0.0075,0.01,0.025]}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(),param,scoring='accuracy')
clf.fit(Xtrain,Ytrain)
print(clf.best_params_)

"""
