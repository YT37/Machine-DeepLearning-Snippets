import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR

dataset = pd.read_csv("PositionSalaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""from sklearn.model_selection import train_test_split
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)"""

from sklearn.preprocessing import StandardScaler

scX = StandardScaler()
X = scX.fit_transform(X)
SCy = StandardScaler()
y = SCy.fit_transform(y.reshape(-1, 1))

regressor = SVR(kernel="rbf")
regressor.fit(X, y)

yPred = SCy.inverse_transform(regressor.predict(scX.transform(np.array([[6.5]]))))


Xgrid = np.arange(min(X), max(X), 0.1)
Xgrid = Xgrid.reshape((len(Xgrid), 1))

plt.scatter(X, y, color="red")
plt.plot(Xgrid, regressor.predict(Xgrid), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
