import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import GridSearchCV

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")
save_path = "/Users/kilhyunkim/Pictures"

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
plot_tree(model)
plt.title("Diabetes Decision Tree")
plt.tight_layout()
plt.savefig(save_path + "/diabetes_decision_tree.png")

# 피쳐 중요도 측정
plot_importance(model)
plt.title("Feature Importance (BaseLine)")
plt.tight_layout()
plt.savefig(save_path + "/feature_importance_baseline.png")

# 모델 재학습하기
cv_params = {
    "max_depth": np.arange(1, 10, 1),
    "learning_rate": np.arange(0.05, 0.6, 0.05),
    "n_estimators": np.arange(50, 300, 10),
}

fix_params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
}

csv = GridSearchCV(XGBClassifier(**fix_params), cv_params, scoring="precision", cv=5, n_jobs=5)
csv.fit(x_train, y_train)
print(csv.best_params_)

y_pred2 = csv.predict(x_test)
predictions = [round(value) for value in y_pred2]

acc2 = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (acc2 * 100))

model2 = XGBClassifier(
    booster="gbtree",\
    objective="binary:logistic",\
    learning_rate=0.03,\
    n_estimators=150,\
    reg_alpha=0.15,\
    reg_lambda=0.7,\
    max_depth=4
)

model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
predictions2 = [round(value) for value in y_pred2]

acc2 = accuracy_score(y_test, predictions2)
print("Accuracy: %.2f%%" % (acc2 * 100))  # 81.82%