import numpy as np
import pandas as pd

import graphviz
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier, plot_tree, plot_importance

from pdpbox import info_plots

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# data = loadtxt("Dataset/pima-indians-diabetes.csv", delimiter=",")
data = pd.read_csv("Dataset/diabetes.csv")

x_data = data[data.columns[0:8]]
y_data = data[data.columns[8]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7)

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

acc = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (acc * 100))

value = [[1, 161, 72, 35, 0, 28.1, 0.527, 20]]  # 입력 형식이 2차원의 ndArray 형식이므로

l = model.predict_proba(value)
print("No diabetes: {:.2f}%\nYes diabetes: {:.2%}".format(l[0][0], l[0][1]))

# Graphviz로 확인하기
rcParams['figure.figsize'] = (12, 10)
plot_tree(model)
plt.show()

# 모델 재학습
model2 = XGBClassifier(max_depth=2)
model2.fit(x_train, y_train)

y_pred2 = model.predict(x_test)
predictions2 = [round(value) for value in y_pred2]

acc2 = accuracy_score(y_test, predictions2)
print("Accuracy: %.2f%%" % (acc2 * 100))

rcParams['figure.figsize'] = (12, 8)
plot_tree(model2, fmap="")
plt.show()

# 피쳐 중요도 측정
rcParams['figure.figsize'] = (10, 10)
plot_importance(model2)
plt.yticks(fontsize=15)
plt.show()

pima_data = data
pima_features = data.columns[0:8]
pima_target = data.columns[8]

fig, axes, summary = info_plots.target_plot(
    data=pima_data
    , feature="Glucose"
    , feature_name="Glucose"
    , target = pima_target
)