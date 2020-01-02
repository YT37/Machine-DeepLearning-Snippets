import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("PositionSalaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""from sklearn.model_selection import train_test_split
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)"""

"""from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
SCy = StandardScaler()
yTrain = SCy.fit_transform(yTrain.reshape(-1,1))"""

regressor = None

yPred = regressor.predict([[6.5]])

Xgrid = np.arange(min(X), max(X), 0.1)
Xgrid = Xgrid.reshape((len(Xgrid), 1))

plt.scatter(X, y, color="red")
plt.plot(Xgrid, regressor.predict(Xgrid), color="blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
