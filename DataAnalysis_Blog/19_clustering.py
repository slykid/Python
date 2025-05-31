import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons

np.random.seed(1)
matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()
plt.show()

# 비교를 위한 KMeans, Agglomerative Clustering 구현
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
kmeans = KMeans(n_clusters=2, random_state=0)
y_pred_kmeans = kmeans.fit_predict(X)
ax1.scatter(X[y_pred_kmeans==0, 0], X[y_pred_kmeans==0, 1], c="lightblue", edgecolor="black", marker="o", s=40, label="Cluster 1")
ax1.scatter(X[y_pred_kmeans==1, 0], X[y_pred_kmeans==1, 1], c="red", edgecolor="black", marker="s", s=40, label="Cluster 2")
ax1.set_title("K-Means Clustering")

ac = AgglomerativeClustering(n_clusters=2, linkage="complete")
y_pred_ac = ac.fit_predict(X)
ax2.scatter(X[y_pred_ac==0, 0], X[y_pred_ac==0, 1], c="lightblue", edgecolor="black", marker="o", s=40, label="Cluster 1")
ax2.scatter(X[y_pred_ac==1, 0], X[y_pred_ac==1, 1], c="red", edgecolor="black", marker="s", s=40, label="Cluster 2")
ax2.set_title("Agglomerative Clustering")

plt.legend()
plt.tight_layout()
plt.show()

# DBSCAN 구현
dbscan = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_pred_db = dbscan.fit_predict(X)
plt.scatter(X[y_pred_db==0, 0], X[y_pred_db==0, 1], c="lightblue", edgecolor="black", marker="o", s=40, label="Cluster 1")
plt.scatter(X[y_pred_db==1, 0], X[y_pred_db==1, 1], c="red", edgecolor="black", marker="s", s=40, label="Cluster 2")

plt.legend()
plt.tight_layout()
plt.show()