import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

matplotlib.use("MacOSX")
plt.style.use("seaborn-v0_8")

# 1. Data load
data = pd.read_csv("Dataset/concrete.csv")
data.info()

# 2. Preprocessing
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# Apply it to every column
data_norm = data.apply(normalize)

print("Normalized 'strength' summary:")
print(data_norm['strength'].describe()[['min', '25%', '50%', 'mean', '75%', 'max']])

print("\nOriginal 'strength' summary:")
print(data['strength'].describe()[['min', '25%', '50%', 'mean', '75%', 'max']])

# 3. Split Train & ReinforcementLearning Dataset
train = data_norm.iloc[0:773].reset_index(drop=True)
test  = data_norm.iloc[773:1030].reset_index(drop=True)

X_train = train.drop(columns='strength')
y_train = train['strength']
X_test  = test.drop(columns='strength')
y_test  = test['strength']

# 4. Modeling
model = MLPRegressor(
    hidden_layer_sizes=(1,),
    activation='logistic',  # similar to R’s default logistic activation
    solver='adam',
    max_iter=2_000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(6,4))
plt.plot(model.loss_curve_, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

corr = np.corrcoef(y_pred, y_test)[0,1]
print(f'Correlation between predictions and true strength: {corr:.4f}')

# 5. Improve model
model2 = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation='logistic',   # sigmoid activation, like R’s neuralnet default
    solver='adam',
    max_iter=2_000,
    random_state=42
)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

plt.figure(figsize=(6,4))
plt.plot(model2.loss_curve_, marker='o')
plt.title('Training Loss Curve (5 Hidden Neurons)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


corr2 = np.corrcoef(y_pred2, y_test)[0,1]
print(f'Correlation between predictions and true strength: {corr2:.4f}')
print(f'Percent correlation: {corr2*100:.2f}%')
