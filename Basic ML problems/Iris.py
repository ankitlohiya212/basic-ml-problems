import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris = load_iris()
X = iris.data
Y= iris.target
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.1)
svm = SVC(kernel = 'rbf' , random_state=0 , gamma=0.2 , C=1.0)
svm.fit(X_train , Y_train)
print ('Training Accuracy:', svm.score(X_train , Y_train))
print ('Test Accuracy:', svm.score(X_test , Y_test))
X= np.transpose(X)
#plt.scatter(X[2],X[3],marker="o")
#plt.show()
