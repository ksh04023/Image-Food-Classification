import numpy as np
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

X_train = np.append(X_train,X_test)
y_train = np.append(y_train,y_test)