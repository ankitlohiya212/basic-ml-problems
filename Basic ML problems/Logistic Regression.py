import numpy as np
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
w1= np.random.rand()
w2= np.random.rand()
w3= np.random.rand()
data = np.loadtxt('ex2data1.txt',delimiter = ',' ,dtype = None)
X = data[:,[0,1]]
Y = data[:,2]
X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
X[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
X[:,2] = (X[:,2]-X[:,2].mean())/X[:,2].std()
Xtrain , Xtest,Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)

def sigmoid(s):
    return 1/(1+np.exp(-s))
def sigmoidprime(s):
    return sigmoid(s)*(1-sigmoid(s))

costs=np.zeros((500,len(Xtrain)),dtype=float)
cost = 0.0
N=[]
N= range(0,500*len(Xtrain))  
for j in range(500):
    for i in range(len(Xtrain)):
        y= w1*Xtrain[i,1] + w2*Xtrain[i,2]+w3*Xtrain[i,0]
        z=sigmoid(y)
        #cost=  -Ytrain[i]*np.log(z) - (1-Ytrain[i])*np.log(1-z)
        #cost = np.square(z-Ytrain[i])
        dcostdw1 = (z-Ytrain[i])*Xtrain[i,1]
        dcostdw2 = (z-Ytrain[i])*Xtrain[i,2]
        dcostdw3 = (z-Ytrain[i])*Xtrain[i,0]
        w1=w1-0.002*dcostdw1
        w2=w2-0.002*dcostdw2
        w3 = w3-0.002*dcostdw3
#        costs[j,i] = cost
#costs = costs.reshape((len(N),1)) 

Ytestpred = np.ones(Ytest.shape)
Ytrainpred = np.ones(Ytrain.shape)
predictions = np.zeros((len(Xtest),1),dtype=float)
for i in range(len(Xtest)):
    ypred = w1*Xtest[i,1] + w2*Xtest[i,2]+w3*Xtest[i,0]
    pred = sigmoid(ypred)
    Ytestpred[i] = pred
    if pred>0.5:
        Ytestpred[i] = 1
    else :
        Ytestpred[i] = 0
for i in range(len(Xtrain)):
    ypred = w1*Xtrain[i,1] + w2*Xtrain[i,2]+w3*Xtrain[i,0]
    pred = sigmoid(ypred)
    Ytrainpred[i] = pred
    if pred>0.5:
        Ytrainpred[i] = 1
    else :
        Ytrainpred[i] = 0

        
trainacc1 = 1-abs(Ytrainpred.reshape(80,)-Ytrain).sum()/80
testacc1 = 1-abs(Ytestpred.reshape(20,)-Ytest).sum()/20
print("Testacc without vectors" ,testacc1)    
print("Trainacc without vectors" ,trainacc1) 

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(Xtrain,Ytrain)
print("Test acc with scikit lib",logreg.score(Xtest,Ytest))
print("Train acc with scikit lib",logreg.score(Xtrain,Ytrain))



theta = np.array([w1,w2,w3])
y = np.matmul(Xtrain,theta)
h = sigmoid(y)
J = -np.matmul(Ytrain.T.reshape(80,1),np.log(h)) - np.matmul((1-Ytrain).T.reshape(80,1),np.log(1-h))
theta = theta.reshape(3,1)-0.02*np.matmul(Xtrain.T,(h.reshape((80,))-Ytrain).T)
Ytrainpred = sigmoid(np.matmul(Xtrain,theta))
Ytestpred =sigmoid(np.matmul(Xtest,theta))
for i in range(len(Ytrainpred)) :
    if float(Ytrainpred[i])>0.5:
        Ytrainpred[i] = 1
    else :
        Ytrainpred[i] = 0
    
for i in range(len(Ytestpred)) :
    if float(Ytestpred[i])>0.5:
        Ytestpred[i] = 1
    else :
        Ytestpred[i] = 0
trainacc = 1-abs(Ytrainpred.reshape(80,)-Ytrain).sum()/80
testacc = 1-abs(Ytestpred.reshape(20,)-Ytest).sum()/20
print("Testacc with vectors" ,testacc)
print("Trainacc with vectors" ,trainacc)



        

