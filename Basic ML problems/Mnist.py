from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

def display(i):
    img = X[i]
    plt.title('Example'+ str(i)+ 'Label:'+str(Y[i])+ 'Predicted:'+str(ypred[i]))
    plt.imshow(img.reshape((28,28)),cmap=plt.cm.gray_r)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
mnist = loadmat('mnist-original')
X , Y = mnist['data'] , mnist['label']
X= X.T
Y = Y.T
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.1,shuffle = True)
X_train , X_val , Y_train , Y_val = train_test_split(X_train,Y_train,test_size=0.2,shuffle = True)

X_train = X_train/255
X_test = X_test/255
X_val = X_val/255
Ytrain = np_utils.to_categorical(Y_train)
Ytest = np_utils.to_categorical(Y_test)
Yval = np_utils.to_categorical(Y_val)

model = Sequential()
model.add(Dense(784,input_shape=(784,),activation='relu',kernel_initializer='normal'))
model.add(Dense(10, activation = 'softmax',kernel_initializer='normal'))
model.compile(loss='categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

history = model.fit(X_train ,Ytrain,batch_size = 512 ,epochs=30,verbose=2, validation_data=(X_val,Yval))

test_accuracy = model.evaluate(x=X_test,y=Ytest,batch_size=200,verbose=2)
print("Test results : ", test_accuracy)

Ypred = model.predict(X)
ypred = []
for i in Ypred:
    ypred.append(np.argmax(i))
    
plot_accuracy(history)
plot_loss(history)

