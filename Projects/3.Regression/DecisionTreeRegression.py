import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv(r"..\1.Datasets\PositionSalaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

yPred = regressor.predict([[6.5]])

Xgrid = np.arange(min(X), max(X), 0.01)
Xgrid = Xgrid.reshape((len(Xgrid), 1))

plt.scatter(X, y, color="red")
plt.plot(Xgrid, regressor.predict(Xgrid), color="blue")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
