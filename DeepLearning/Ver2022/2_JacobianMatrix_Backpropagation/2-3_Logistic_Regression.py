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
y = (X > decision_boundary).astype(np.int)

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

    break



