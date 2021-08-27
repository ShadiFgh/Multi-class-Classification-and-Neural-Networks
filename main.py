import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('data.mat')
x = data['X']
y = data['y']


weights = scipy.io.loadmat('weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

# print(np.insert(x[0], 0, 1))

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

def forward_propagation(theta1, theta2, x):
    HP = 1 / (1 + np.exp(- np.matmul(x, theta1.T)))
    # for i in range(0, len(HP)):
    #     if HP[i] < 0.5:
    #         HP[i] = 0
    #     elif HP[i] > 0.5:
    #         HP[i] = 1
    pp = 1 / (1 + np.exp(- np.matmul(np.insert(HP, 0, 1), theta2.T)))
    for i in range(0, len(pp)):
        if pp[i] < 0.5:
            pp[i] = 0
        elif pp[i] > 0.5:
            pp[i] = 1
    return pp

def predict_y():
    predicted_y = np.zeros((x.shape[0], 1))
    for i in range(1, len(np.unique(y))+1):
        # y_hat = []
        for j in range(0, len(y)):
            if forward_propagation(theta1, theta2, np.insert(x[j], 0, 1))[i - 1] == 1:
                # y_hat.append(1)
                predicted_y[j] = i
            # else:
            #     y_hat.append(0)
        # print(y_hat)
    return predicted_y

print(predict_y())




# print(forward_propagation(theta1, theta2, np.insert(x[0], 0, 1)))

def cost(theta1, theta2, x, y):
    diff = (y * (np.log(forward_propagation(theta1, theta2, x)))) + ((1 - y) * (np.log(1 - forward_propagation(theta1, theta2, x))))
    total = sum(diff)
    return total / (-x.shape[0])

# print(cost(theta1, theta2, x, y))