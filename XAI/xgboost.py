from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

data = loadtxt("Dataset/pima-indians-diabetes.csv", delimiter=",")

x_data = data[:, 0:8]
y_data = data[:, 8]

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