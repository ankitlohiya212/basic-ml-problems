import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

data = np.loadtxt('ex2data2.txt',delimiter = ',' ,dtype = None)
X = data[:,[0,1]]
Y = data[:,2]
Xtrain , Xtest,Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)
clf = svm.SVC(kernel='rbf')
clf.fit(Xtrain,Ytrain)
pred = clf.predict(Xtest)
print(1-sum(abs(pred-Ytest))/len(Xtest))
predtrain = clf.predict(Xtrain)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
