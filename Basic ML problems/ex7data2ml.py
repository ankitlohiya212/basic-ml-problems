from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
data = loadmat('ex7data2.mat')
X= data['X']
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
Xpred = kmeans.predict(X)
plt.scatter(X[:,0],X[:,1],c=Xpred)
plt.show()
"""

data = loadmat('ex8_movieParams.mat')
data1 = loadmat('ex8_movies.mat')
R = data1['R']
Y = data1['Y']
X = data['X']
Theta = data['Theta']
num_movies = data['num_movies'][0][0]
num_users = data['num_users'][0][0]
num_features = data['num_features'][0][0]


