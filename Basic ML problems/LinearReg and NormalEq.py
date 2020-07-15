import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score, mean_absolute_error

f=open('ex1data1.txt','r')
data = f.read()
data = np.matrix(data)
data = data.reshape(97,2)
X = data[:,0]
Y = data[:,1]

data = np.append(np.ones((97,1),dtype=int),X,axis=1)
Xtrain = data[0:70]
Xtrain[:,1] = (Xtrain[:,1] - Xtrain[:,1].mean())/Xtrain[:,1].std()
Ytrain = Y[:70]

Xtest = data[70:]
Xtest[:,1] = (Xtest[:,1]-Xtest[:,1].mean())/Xtest[:,1].std()
Ytest = Y[70:]

def train_with_iteration(X, Y):
    start = time.time()
    w1= np.random.rand()
    w2= np.random.rand()
    b= np.random.rand()
    ITERATIONS = 150
    costs=np.zeros((ITERATIONS,1),dtype=float)
    N=[]
    N= range(0,ITERATIONS)  
    for j in range(ITERATIONS):
        for i in range(len(X)):
            y= w1*X[i,0] + w2*X[i,1]
            cost=  np.square(y - Y[i])
            dcostdw1 = 2*(y-Y[i])
            dcostdw2 = 2*(y-Y[i])*X[i,1]
            w1=w1-0.0002*dcostdw1
            w2=w2-0.0002*dcostdw2
        costs[j] = cost
    end = time.time()
    time_iteration = end-start
    #print("Time taken for iteration = ", time_iteration)
    #plt.plot(N,costs)
    return w1,w2
    plt.show()

def predict_with_iteration(Xtest, w1, w2):    
    predictions = np.zeros((len(Xtest),1),dtype=float)
    for i in range(len(Xtest)):
        pred = w1*Xtest[i,0] + w2*Xtest[i,1]
        predictions[i,:] = (pred)
    return predictions



def train_with_matrices(X,Y):
    theta= np.dot(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
    return theta

def predict_with_matrices(Xtest,theta):
    Prediction =np.matmul(Xtest,theta)
    return Prediction

def train_and_predict_with_library(Xtrain,Ytrain, Xtest):
    lr = LinearRegression()
    lr.fit(Xtrain, Ytrain)
    p = lr.predict(Xtest)
    return p


"""
w1,w2 = train_with_iteration(Xtrain,Ytrain)
Pred_with_iteration = predict_with_iteration(Xtest, w1, w2)

theta = train_with_matrices(Xtrain, Ytrain)
Pred_with_matrices = predict_with_matrices(Xtest, theta)

Pred_with_library = train_and_predict_with_library(Xtrain,Ytrain, Xtest)



print("\nAccuracy between library and matmul" ,r2_score(Pred_with_library, Pred_with_matrices))
#print("Time taken for Matmul :", time2 )
print("Accuracy between library and Iterative proces" ,r2_score(Pred_with_library, Pred_with_iteration))
#print("\nTime taken for Iterative process :", time1 )


"""

xtrain = np.arange(1,7)
xtrain = xtrain.reshape(len(xtrain),1)
xtrain = np.append(np.ones((len(xtrain),1),dtype=int),xtrain ,axis=1)
ytrain = np.arange(1,7)
ytrain = ytrain.reshape(len(ytrain),1)
xtest = np.arange(50,55)
xtest = xtest.reshape(len(xtest),1)
xtest = np.append(np.ones((len(xtest),1),dtype=int),xtest ,axis=1)

w1,w2 = train_with_iteration(xtrain, ytrain)
Pred_with_iteration = predict_with_iteration(xtest, w1, w2)
print("\nIteration : ", Pred_with_iteration)

theta = train_with_matrices(xtrain, ytrain)
Pred_with_matrices = predict_with_matrices(xtest, theta)
print("\nMatrices : ", Pred_with_matrices)

Pred_with_library = train_and_predict_with_library(xtrain,ytrain, xtest)
print("\nLibrary : ", Pred_with_library)
