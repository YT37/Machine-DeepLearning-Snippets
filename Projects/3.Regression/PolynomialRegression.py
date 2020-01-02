import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

polyReg = PolynomialFeatures(degree=4)
Xpoly = polyReg.fit_transform(X)
polyReg.fit(Xpoly, y)

linReg2 = LinearRegression()
linReg2.fit(Xpoly, y)

linReg2.predict(polyReg.fit_transform([[6.5]]))

Xgrid = np.arange(min(X), max(X), 0.1)
Xgrid = Xgrid.reshape((len(Xgrid), 1))

plt.scatter(X, y, color="red")
plt.plot(Xgrid, linReg2.predict(polyReg.fit_transform(Xgrid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
