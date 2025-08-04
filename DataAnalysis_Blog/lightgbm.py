import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics._classification import confusion_matrix

from lightgbm import LGBMClassifier, early_stopping

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")
save_path = "/Users/kilhyunkim/Pictures"

def get_clf_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Confusion Matrix")
    print(cm)
    print("Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}".format(accuracy, precision, recall))

dataset = load_breast_cancer()
feature = dataset.data
target = dataset.target

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=156)

lgbm_wrapper = LGBMClassifier(n_estimators=400)

evals = [(X_test, y_test)]
lgbm_wrapper.fit(X_train, y_train, eval_metric="logloss", eval_set=evals)
y_pred = lgbm_wrapper.predict(X_test)

get_clf_eval(y_test, y_pred)
