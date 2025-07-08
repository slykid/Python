import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use('seaborn-v0_8')

# set params
N, n_features = 1000, 3
learning_rate = 0.01
epochs = 100
batch_size = 32

t_W = np.random.uniform(-1, 1, (n_features, 1))
t_b = np.random.uniform(-1, 1, (1, 1))

W = np.random.uniform(-1, 1, (n_features, 1))
b = np.random.uniform(-1, 1, (1, 1))

n_batch = N // batch_size

# generate dataset
x_data = np.random.normal(0, 1, (N, n_features))
y_data = x_data @ t_W + t_b
y_data = (y_data > 0).astype(np.int64)

# training
J_track = []
acc_track = []
for epoch in range(epochs):
    for batch in range(n_batch):

        # get mini-batches
        X = x_data[batch * batch_size : (batch + 1) * batch_size, ...]
        y = y_data[batch * batch_size : (batch + 1) * batch_size, ...]

        # forward propagation
        Z = X @ W + b
        y_pred = 1 / (1 + np.exp(-Z))
        J0 = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        J = np.mean(J0)
        J_track.append(J)

        # Calculate Accuracy
        _y_pred = (y_pred > 0.5).astype(np.int64)
        n_correct = np.sum((_y_pred == y).astype(np.int64))
        acc = n_correct / batch_size
        acc_track.append(acc)

        # jacobians
        dJ_dJ0 = 1 / N * np.ones((1, batch_size))
        dJ0_dpred = np.diag(((y_pred - y) / (y_pred * (1 - y_pred))).flatten())
        dpred_dZ = np.diag((y_pred * (1 - y_pred)).flatten())
        dZ_dW = X
        dZ_db = np.ones((batch_size, 1))

        # back propagation
        dJ_dpred = dJ_dJ0 @ dJ0_dpred
        dJ_dZ = dJ_dpred @ dpred_dZ
        dJ_dW = dJ_dZ @ dZ_dW
        dJ_db = dJ_dZ @ dZ_db

        # parameter update
        W = W - learning_rate * dJ_dW.T
        b = b - learning_rate * dJ_db

fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes[0].plot(J_track)
axes[1].plot(acc_track)
plt.show()


