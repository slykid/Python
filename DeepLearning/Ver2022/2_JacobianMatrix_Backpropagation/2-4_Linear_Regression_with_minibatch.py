import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from sympy.physics.units import length

np.random.seed(0)
matplotlib.use("MacOSX")
plt.style.use('seaborn-v0_8')

# Set Params
N, n_features = 300, 5
learning_rate = 0.03
t_W = np.random.uniform(-1, 1, (n_features, 1))
t_b = np.random.uniform(-1, 1, (1, 1))

W = np.random.uniform(-1, 1, (n_features, 1))
b = np.random.uniform(-1, 1, (1, 1))

epochs = 100
batch_size = 32

# Generate dataset
x_data = np.random.randn(N, n_features)
y_data = x_data @ t_W + t_b

J_track = []
W_track = []
b_track = []

n_batch = N // batch_size

for epoch in range(epochs):
    for idx in range(n_batch):
        W_track.append(W)
        b_track.append(b)

        X = x_data[idx * batch_size : (idx + 1) * batch_size, ...]
        Y = y_data[idx * batch_size : (idx + 1) * batch_size, ...]

        # Forward Propagation
        y_pred = X @ W + b
        J0 = (Y - y_pred) ** 2
        J = np.mean(J0)
        J_track.append(J)

        # jacobians
        dJ_dJ0 = 1/batch_size * np.ones((1, batch_size))
        dJ0_dpred = np.diag(-2 * (Y - y_pred).flatten())
        dpred_dW = X                                           # 본래 X^T 지만, 파이썬에서는 transpose 가 적용된 상태임
        dpred_db = np.ones((batch_size, 1))

        # Back Propagation
        dJ_dpred = dJ_dJ0 @ dJ0_dpred
        dJ_dW = dJ_dpred @ dpred_dW
        dJ_db = dJ_dpred @ dpred_db

        # Parameter Update
        W = W - learning_rate * dJ_dW.T
        b = b - learning_rate * dJ_db

W_track = np.hstack(W_track)
b_track = np.concatenate(b_track)

cmap = cm.get_cmap('tab10')
fig, axes = plt.subplots(2, 1, figsize=(20, 15))
axes[0].plot(J_track)

for w_idx, (t_w, w_track) in enumerate(zip(t_W, W_track)):
    axes[1].axhline(y=t_w, linestyle=':', color=cmap(w_idx))
    axes[1].plot(w_track, color=cmap(w_idx))
axes[1].axhline(y=t_b, linestyle=':', color='black')
axes[1].plot(b_track, color='black')
plt.show()