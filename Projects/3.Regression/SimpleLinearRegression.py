import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("SalaryData.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=1 / 3, random_state=0)

regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

plt.scatter(Xtrain, yTrain, color="red")
plt.plot(Xtrain, regressor.predict(Xtrain), color="blue")
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(Xtest, yTest, color="red")
plt.plot(Xtrain, regressor.predict(Xtrain), color="blue")
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()
