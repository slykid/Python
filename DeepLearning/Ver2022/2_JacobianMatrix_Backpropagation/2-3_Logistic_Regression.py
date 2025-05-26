import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib import cm as cm

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# Set Parameter
N = 1000
learning_rate = 0.03
t_w = np.random.uniform(-3, 3, (1, ))
t_b = np.random.uniform(-3, 3, (1, ))

w = np.random.uniform(-3, 3, (1, ))
b = np.random.uniform(-3, 3, (1, ))

# Generate Dataset
decision_boundary = -t_b/t_w

x_data = np.random.normal(decision_boundary, 1, size=(N, ))
y_data = x_data * t_w + t_b
y_data = (x_data > decision_boundary).astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_data, y_data)
plt.show()

x_range = np.linspace(x_data.min(), x_data.max(), N)
cmap = plt.get_cmap("rainbow", lut=N)

J_track, w_track, b_track = list(), list(), list()

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    w_track.append(w)
    b_track.append(b)

    # visualize updated model
    y_range = w * x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    # ax.plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    # plt.show()

    # Forward Propagation
    z = x * w + b
    y_pred = 1 / (1 + np.exp(-z))
    J = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    J_track.append(J)

    # Jacobian
    dJ_dpred = (y_pred - y) / (y_pred * (1 - y_pred))
    dpred_dz = y_pred * (1 - y_pred)
    dz_dw = x
    dz_db = 1

    # Back Propagation
    dJ_dz = dJ_dpred * dpred_dz
    dJ_dw = dJ_dz * dz_dw
    dJ_db = dJ_dz * dz_db

    # Parameter Update
    w = w - learning_rate * dJ_dw
    b = b - learning_rate * dJ_db

# Visualize loss
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(J_track)
axes[0].set_ylabel("BCCE", fontsize=15)
axes[0].tick_params(labelsize=15)

axes[1].axhline(y=t_w, color="darkred", linestyle=":")
axes[1].plot([element for array in w_track for element in array], color="darkred")
axes[1].axhline(y=t_b, color="darkblue", linestyle=":")
axes[1].plot([element for array in b_track for element in array], color="darkblue")
axes[1].tick_params(labelsize=15)

plt.show()

# Decision Boundary 시각화
# - Decision Boundary를 지키면서 학습을 한다는 사실 확인 가능
arr_w_track = np.array(w_track)
arr_b_track = np.array(b_track)

arr_db_track = -arr_b_track/arr_w_track
db = -t_b/t_w

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(y=db, color="black", linestyle=":")
ax.plot(arr_w_track, color="black")
plt.show()
