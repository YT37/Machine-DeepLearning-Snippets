# Linear Regression
"""from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)"""

# Multiple Linear Regression
"""from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

Xopt = X[:, [0, 1, 2, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 1, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 3, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()"""

# Support Vector Regression
"""from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

yPred = regressor.predict(Xtest)"""

# Polynomial Regression
"""from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=4)
Xpoly = polyReg.fit_transform(X)
polyReg.fit(Xpoly, y)

from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(Xpoly, y)

linReg.predict(polyReg.fit_transform(Xtest))"""

# Decision Tree Regression
"""from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

yPred = regressor.predict(Xtest)"""

# Random Forest Regression
"""from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

yPred = regressor.predict(Xtest)"""