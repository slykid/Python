import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import cm as cm

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# Set Parameter
N = 300
learning_rate = 0.01
t_w = np.random.uniform(-3, 3, (1, ))
t_b = np.random.uniform(-3, 3, (1, ))

w = np.random.uniform(-3, 3, (1, ))
b = np.random.uniform(-3, 3, (1, ))

# Generate Dataset
decision_boundary = -t_b/t_w

X = np.random.normal(decision_boundary, 1, size=(N, ))
y = X * t_w + t_b
y = (X > decision_boundary).astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, y)
plt.show()

x_range = np.linspace(X.min(), X.max(), N)
cmap = plt.get_cmap("rainbow", lut=N)

J_track, w_track, b_track = list(), list(), list()
for idx, (_x, _y) in enumerate(zip(X, y)):
    w_track.append(w)
    b_track.append(b)

    # visualize updated model
    y_range = w * x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    # ax.plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    # plt.show()

    # Forward Propagation
    z = _x * w + b
    y_pred = 1 / (1 + np.exp(-z))
    J = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    J_track.append(J)

    # Jacobian
    dJ_dpred = (y_pred - y) / (y_pred * (1 - y_pred))
    dpred_dz = y_pred * (1 - y_pred)
    dz_dw = _x
    dz_db = 1

    # Back Propagation
    dJ_dz = dJ_dpred * dpred_dz
    dJ_dw = dJ_dz * dz_dw
    dJ_db = dJ_dz * dz_db

    # Parameter Update
    w = w - learning_rate * dJ_dw
    b = b - learning_rate * dJ_db

# Visualize Loss
fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes[0].plot(J_track)
axes[0].set_ylabel("BCCE", fontsize=30)
axes[0].tick_params(labelsize=30)

axes[1].axhline(y=t_w[0], color="darkred", linestyle=":")
axes[1].plot(np.concatenate(w_track).tolist(), color="darkred")
axes[1].axhline(y=t_b[0], color="darkblue", linestyle=":")
axes[1].plot(np.concatenate(b_track).tolist(), color="darkblue")

plt.show()

# With N-Features
N, n_features = 1000, 3
learning_rate = 0.03

t_W = np.random.uniform(-1, 1, (n_features, 1))
t_b = np.random.uniform(-1, 1, (1, ))


W = np.random.uniform(-1, 1, (n_features, 1))
b = np.random.uniform(-1, 1, (1, ))

# Generate Dataset
x_data = np.random.randn(N, n_features)
y_data = x_data @ t_W + t_b
y_data = 1 / (1 + np.exp(-y_data))
y_data = (y_data > 0.5).astype(int)

J_track, acc_track = list(), list()
n_correct = 0
for idx, (X, y) in enumerate(zip(x_data, y_data)):

    # Feed forward progagation
    z = X @ W + b
    pred = 1 / (1 + np.exp(-z))
    J =  -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    J_track.append(J.squeeze())

    # Calculate Accuracy
    _pred = (pred > 0.5).astype(int).squeeze()
    if(_pred == y):
        n_correct += 1

    acc_track.append(n_correct / (idx + 1))

    # Jacobian
    dJ_dpred = (pred - y) / (pred * (1 - pred))
    dpred_dz = pred * (1 - pred)
    dz_dW = X.reshape(1, -1)
    dz_db = 1


    # Back Propagation
    dJ_dz = dJ_dpred * dpred_dz
    dJ_dW = dJ_dz * dz_dW
    dJ_db = dJ_dz * dz_db

    # Parameter Update
    W = W - learning_rate * dJ_dW.T
    b = b - learning_rate * dJ_db

# Visualize Loss
fig, axes = plt.subplots(2, 1,figsize=(20, 10))
axes[0].plot(J_track)
axes[1].plot(acc_track)
axes[0].set_ylabel('BCEE', fontsize=30)
axes[0].tick_params(labelsize=20)
axes[1].set_ylabel('Accumulated Accuracy', fontsize=30)
axes[1].tick_params(labelsize=20)

plt.show()