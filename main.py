import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = scipy.io.loadmat('data.mat')

x = data['X']
y = data['y']
n = x.shape[1] + 1
m = x.shape[0]
normalized_x = x / 255
x0 = np.ones((m, 1))
x = np.hstack((x0, x))

weights = scipy.io.loadmat('weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

learning_rate = 0.0005



def h(coefs, x):
    HP = 1 / (1 + np.exp(- np.matmul(x, coefs.T)))
    return HP


def cost(coefs, x, y):
    diff = (y * (np.log(h(coefs, x)))) + ((1 - y) * (np.log(1 - h(coefs, x))))
    total = sum(diff)
    return total / (-m)


def theta(coefs):
    diff = h(coefs, x) - y
    mul = diff * x
    total = sum(mul)
    coefs = coefs - (learning_rate * total)

    return coefs

def find_y_hat(i):
     y_hat = np.zeros((x.shape[0], 1))
     for j in range(0, len(y)):
        if y[j] == i:
               y_hat[j] = 1
     return y_hat

c = []
lr = []

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
for j in range(1, len(np.unique(y))+1):
    coefs = np.random.randn(1, n)
    y1 = find_y_hat(j)
    for i in range(1000):
        if i % 10 == 0:
            learning_rate *= 0.995
            lr.append(learning_rate)
        coefs = theta(coefs)
        co = cost(coefs, x, y1)[0]
        print(f"LR: {learning_rate} Lv: {i + 1} ==> Cost: {co}")
        c.append(co)


for i in range(1, len(np.unique(y))+1):
    predicted_y = np.zeros((x.shape[0], 1))
    coefs = np.random.randn(1, n)
    for j in range(0, len(y)):
        if h(coefs, x)[j] > 0.5 :
            predicted_y[j] = 1
    print(predicted_y)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.plot(c)
plt.show()


def forward_propagation(theta1, theta2, x):
    layer2 = 1 / (1 + np.exp(- np.matmul(x, theta1.T)))
    layer3 = 1 / (1 + np.exp(- np.matmul(np.insert(layer2, 0, 1), theta2.T)))
    for i in range(0, len(layer3)):
        if layer3[i] < 0.5:
            layer3[i] = 0
        elif layer3[i] > 0.5:
            layer3[i] = 1
    return layer3
for i in range(0, len(y)):
    print(forward_propagation(theta1, theta2, x[i]))

def predict_y():
    predicted_y = np.zeros((x.shape[0], 1))
    for i in range(1, len(np.unique(y))+1):
        # y_hat = []
        for j in range(0, len(y)):
            if forward_propagation(theta1, theta2, x[j])[i - 1] == 1:
                # y_hat.append(1)
                predicted_y[j] = i
            # else:
            #     y_hat.append(0)
        # print(y_hat)
    return predicted_y

print(predict_y())



""" plot the pixles"""
# X = []
# for i in range(0, 5000):
#     X.append(np.reshape(x[i][:400], (20, 20)))
# print(np.array(X).shape)

# pixel_plot = plt.figure()
# pixel_plot.add_axes()
# plt.title("pixel_plot")
# pixel_plot = plt.imshow(X[0], cmap='twilight', interpolation='nearest')
# plt.colorbar(pixel_plot)
# plt.savefig('pixel_plot.png')
# plt.gray()
# plt.show()