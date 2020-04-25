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
