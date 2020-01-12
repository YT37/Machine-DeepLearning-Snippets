import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(r"../1.Datasets/PositionSalaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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
