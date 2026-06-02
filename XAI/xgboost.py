import numpy as np
import pandas as pd
from pdpbox.pdp import PDPIsolate

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

from pdpbox import info_plots, pdp

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

# InfoPlot 생성
# - 참고자료: https://pdpbox.readthedocs.io/en/latest/PDPIsolate.html
# - 예시: https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb
target_plot = info_plots.TargetPlot(df=data, feature="Glucose", feature_name="Glucose", target="Outcome")
fig, axes, summary_df = target_plot.plot(
    figsize=(20, 10),
    ncols=2,
    plot_params=None,
    engine='matplotlib',
)

plt.savefig(save_path + "/target_plot_glucose.png")

target_plot = info_plots.TargetPlot(df=data, feature="BloodPressure", feature_name="BloodPressure", target="Outcome")
fig, axes, summary_df = target_plot.plot(
    figsize=(20, 10),
    ncols=2,
    plot_params=None,
    engine='matplotlib',
)

plt.savefig(save_path + "/target_plot_bloodpressure.png")

predict_plot = info_plots.PredictPlot(model=model, df=data, model_features=data.columns[:8], feature="Glucose", feature_name="Glucose")
fig, axes, summary_df = predict_plot.plot(
    figsize=(20, 10),
    ncols=2,
    plot_params=None,
    engine='matplotlib',
)

plt.savefig(save_path + "/predict_plot_glucose.png")

pdp_gc = pdp.PDPIsolate(model=model, df=data, model_features=data.columns[:8], feature="Glucose", feature_name="Glucose")
fig, axes = pdp_gc.plot(
    plot_lines=True,
    frac_to_plot=0.5,
    plot_pts_dist=True,
    engine='matplotlib',
    template='plotly_white',
)
plt.savefig(save_path + "/isolate_plot_glucose.png")

target_plot_interact = info_plots.InteractTargetPlot(df=data, features=["BloodPressure", "Glucose"], feature_names=["BloodPressure", "Glucose"], target="Outcome")
fig, axes, summary_df = target_plot_interact.plot(
    figsize=(20, 10),
    ncols=2,
    engine='matplotlib',
    template='plotly_white',
)
plt.savefig(save_path + "/interact_plot_glucose_bloodpressure.png")

pdp_interaction = pdp.PDPInteract(model=model, df=data, model_features=data.columns[:8], features=["BloodPressure", "Glucose"], feature_names=["BloodPressure", "Glucose"])
fig, axes = pdp_interaction.plot(
    plot_type="contour",
    plot_pdp=True,
    show_percentile=True,
    engine='matplotlib',
    template='plotly_white',
)
plt.savefig(save_path + "/interaction_plot_glucose_bloodpressure.png")

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

# 모델 평가
## confusion matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)

def plot_confusion_matrix(cm, classes, save_path,normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{save_path}/confusion_matrix.png")

def show_data(cm, print_res=0):
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]

    if print_res:
        print("Precision =        {:.3f}".format(tp / (tp + fp)))
        print("Recall (TPR) =     {:.3f}".format(tp / (tp + fn)))
        print("Fallout (FPR) =    {:.3f}".format(fp / (fp + tn)))

    return tp/(tp + fp), tp/(tp + fn), fp / (fp + tn)


plot_confusion_matrix(cm, ['0', '1'], save_path)
show_data(cm, print_res=1)
# Precision =        0.796
# Recall (TPR) =     0.684
# Fallout (FPR) =    0.103