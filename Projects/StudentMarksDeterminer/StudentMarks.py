import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

dataset = pd.read_csv(r"../1.Datasets/StudentMath.csv", delimiter=";")

dataset = dataset.drop(dataset.loc[:, "address":'Walc'].columns, axis=1)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.3015,
                                                random_state=0)

regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = np.around(regressor.predict(Xtest), decimals=0)

X = np.append(arr=np.ones((395, 1)).astype(int), values=X, axis=1)

rmse = np.sqrt(((yPred - yTest) ** 2).mean())

r2 = r2_score(yTest, yPred)
