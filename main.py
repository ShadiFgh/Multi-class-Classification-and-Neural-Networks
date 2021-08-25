import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('data.mat')
x = data['X']
y = data['y']

X = []

for i in range(0, 5000):
    X.append(np.reshape(x[i][:400], (20, 20)))
# print(np.array(X).shape)

pixel_plot = plt.figure()
pixel_plot.add_axes()
plt.title("pixel_plot")
pixel_plot = plt.imshow(X[999], cmap='twilight', interpolation='nearest')
plt.colorbar(pixel_plot)
plt.savefig('pixel_plot.png')
plt.gray()
plt.show()

