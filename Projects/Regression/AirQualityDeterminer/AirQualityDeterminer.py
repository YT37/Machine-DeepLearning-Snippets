import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("AirQuality.csv")

X = dataset.iloc[:43000, [6, 7, 8, 10, 11, 12]].values
y = dataset.iloc[:43000, 5].values

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=0)

scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
SCy = StandardScaler()
yTest = SCy.fit_transform(yTest.reshape(-1, 1))
yTrain = SCy.fit_transform(yTrain.reshape(-1, 1))

regressor = RandomForestRegressor(n_estimators=250, random_state=0)
regressor.fit(Xtest, yTest)

yPred = regressor.predict(Xtest)

yPredcm = np.zeros_like(yPred)
yPredcm[yPred > 0.6] = 1
yPredcm[yPred < 0.6] = 0

yTestcm = np.zeros_like(yTest)
yTestcm[yTest > 0.6] = 1
yTestcm[yTest < 0.6] = 0

cm = confusion_matrix(yTestcm, yPredcm)

accuracy = accuracy_score(yTestcm, yPredcm)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3
