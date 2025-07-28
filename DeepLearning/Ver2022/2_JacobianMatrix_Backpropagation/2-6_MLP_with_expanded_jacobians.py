import numpy as np
from numpy.random import normal
from numpy import zeros

import matplotlib
from matplotlib import pyplot as plt
from termcolor import colored

from tensorflow.keras.datasets.mnist import load_data

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use('seaborn-v0_8')
save_fig_path = "/Users/kilhyunkim/Pictures"


# load images
(train_images, train_labels), test_ds = load_data()

print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape)

view_images = train_images[:9, ...]
print(view_images.shape)

fig, axes  = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for ax_idx, ax in enumerate(axes.flat):
    image = view_images[ax_idx]
    ax.imshow(image)

# set test env.
n_data = train_images.shape[0]
n_features = train_images.shape[1] * train_images.shape[2]
batch_size = 64
n_batches = n_data // batch_size
epochs = 20
learning_rate = 0.03
units = [64, 32, 10]

# initialize w, b
W1 = normal(0, 1, (n_features, units[0]))
B1 = zeros(units[0])

W2 = normal(0, 1, (units[0], units[1]))
B2 = zeros(units[1])

W3 = normal(0, 1, (units[1], units[2]))
B3 = zeros(units[2])

print(colored("W/B shapes", "green"))
print(f"W1/B1: {W1.shape}/{B1.shape}")
print(f"W2/B2: {W2.shape}/{B2.shape}")
print(f"W3/B3: {W3.shape}/{B3.shape}")

# training
losses, accs = list(), list()
for epoch in range(epochs):
    n_correct, n_data = 0, 0
    for batch_idx in range(n_batches):
        # get mini-batch
        images = train_images[batch_idx * batch_size:(batch_idx + 1) * batch_size, ...]
        X = images.reshape(batch_size, -1) / 255.
        Y = train_labels[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # forward propagation
        ## dense1
        Z1 = X @ W1 + B1
        A1 = 1 / (1 + np.exp(-Z1))

        ## dense2
        Z2 = A1 @ W2 + B2
        A2 = 1 / (1 + np.exp(-Z2))

        ## dense3
        L = A2 @ W3 + B3
        y_pred = np.exp(L) / np.sum(np.exp(L), axis=1, keepdims=True)

        ## Loss
        J = np.mean(-np.log(y_pred[np.arange(batch_size), Y]))
        losses.append(J)

        # calculate accuracy
        y_pred_label = np.argmax(y_pred, axis=1)
        n_correct += np.sum(y_pred_label == Y)
        n_data += batch_size

        # Back Propagation
        labels = Y.copy()
        Y = np.zeros_like(y_pred)
        Y[np.arange(batch_size), labels] = 1

        ## loss
        dL = -1 / batch_size * (Y - y_pred)

        ## dense3
        dA2 = dL @ W3.T
        dW3 = A2.T @ dL
        dB3 = np.sum(dL, axis=0)

        ## dense2
        dZ2 = dA2 * A2 * (1 - A2)
        dA1 = dZ2 @ W2.T
        dW2 = A1.T @ dZ2
        dB2 = np.sum(dZ2, axis=0)

        ## dense1
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = X.T @ dZ1
        dB1 = np.sum(dZ1, axis=0)

        ## parameter update
        W3, B3 = W3 - learning_rate * dW3, B3 - learning_rate * dB3
        W2, B2 = W2 - learning_rate * dW2, B2 - learning_rate * dB2
        W1, B1 = W1 - learning_rate * dW1, B1 - learning_rate * dB1

    accs.append(n_correct / n_data)

# visualize result
fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes[0].plot(losses)
axes[1].plot(accs)

axes[0].set_title("Train Loss", color="darkblue", fontsize=40)
axes[0].set_xlabel("Iter Idx", fontsize=30)
axes[0].set_ylabel("CCEE", fontsize=30)

axes[1].set_title("Train Accuracy", color="darkblue", fontsize=40)
axes[1].set_xlabel("Epoch", fontsize=30)
axes[1].set_ylabel("Accuracy", fontsize=30)
axes[1].set_yticks(np.linspace(0.4, 1.0, 7))

axes[0].tick_params(labelsize=30)
axes[1].tick_params(labelsize=30)

fig.tight_layout()
plt.savefig(save_fig_path + "/MLP_with_expanded_jacobian.png")

