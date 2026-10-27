import numpy as np
import pandas as pd

data_train = pd.read_csv('archive/train.csv')
data_test = pd.read_csv('archive/test.csv')

data_train = np.array(data_train)
data_test = np.array(data_test)

m, n = data_train.shape
X_train = data_train[:, 1:n] / 255.0
y_train = data_train[:, 0]

X_test = data_test[:, 1:n] / 255.0
y_test = data_test[:, 0]

m, n = X_train.shape

W1 = np.random.rand(n, 10) - 0.5
b1 = np.random.rand(1, 10) - 0.5
W2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(1, 10) - 0.5

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def one_hot(y):
    Y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        Y[i][y[i]] = 1
    return Y

def for_prob(W1, b1, W2, b2, X):
    Z1 = X @ W1 + b1
    a1 = ReLU(Z1)
    Z2 = a1 @ W2 + b2
    a2 = softmax(Z2) # both softmax and sigmoid will work
    return Z1, a1, Z2, a2

def back_prop(W1, b1, W2, b2, Z1, a1, Z2, a2, X, y):
    m = X.shape[0]
    Y = one_hot(y)
    
    dZ2 = a2 - Y
    dW2 = (1 / m) * np.dot(a1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    
    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0) 
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

def gradient_descent(alpha, iterations, W1, b1, W2, b2, X, y):
    for i in range(iterations):
        Z1, a1, Z2, a2 = for_prob(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(W1, b1, W2, b2, Z1, a1, Z2, a2, X, y)
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        if i % 50 == 0:
            predictions = np.argmax(a2, axis=1)
            acc = np.mean(predictions == y)
            print(f"Iteration {i} | Training Accuracy: {acc:.4f}")
            
    return W1, b1, W2, b2

print("Training started...")
W1, b1, W2, b2 = gradient_descent(0.1, 1000, W1, b1, W2, b2, X_train, y_train)
