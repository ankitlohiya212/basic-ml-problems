import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA


df = pd.read_csv('winequality-red.csv' , encoding = 'latin-1')

corr = df.corr()
for i in df.columns:
    if abs(corr['quality'][i]) < 0.25 :
        df = df.drop(i, axis = 1)
        
pca = PCA(n_components=3)
X = pca.fit_transform(df)

df1 = pd.DataFrame(data = X , columns = ['PC1', 'PC2','PC3'])
df1['Y'] = df['quality'].values

corr = df1.corr()

for i in df1.columns:
    if abs(corr['Y'][i]) < 0.25 :
        df1 = df1.drop(i, axis = 1)
        
X = df1.values[:,:2]
Y = df1['Y'].values
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
Xtrain, Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(Xtrain,Ytrain)

Ypred = np.round(regr.predict(Xtest))
Ytrain_pred = np.round(regr.predict(Xtrain))

Ytest = Ytest.ravel()
Ytest = np.transpose(Ytest)
Ytrain = Ytrain.ravel()
Ytrain = np.transpose(Ytrain)

print("Test Error = " , mean_squared_error(Ytest,Ypred))
print("Test Score = ", r2_score(Ytest,Ypred))
print()
print("Train Error = " , mean_squared_error(Ytrain,Ytrain_pred))
print("Train Score = ", r2_score(Ytrain,Ytrain_pred))

