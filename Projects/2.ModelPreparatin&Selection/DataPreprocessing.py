import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Projects/1.Datasets/Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

"""from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
SCy = StandardScaler()
yTrain = SCy.fit_transform(yTrain.reshape(-1,1))"""
