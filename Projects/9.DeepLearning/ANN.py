import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

dataset = pd.read_csv(r"../1.Datasets/ChurnModelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

ct = ColumnTransformer(
    [('Gender', OneHotEncoder(categories="auto"), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer(
    [('Country', OneHotEncoder(categories="auto"), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X.astype(float)
X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(
    X, y, test_size=0.1, random_state=0)

scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)

classifier = Sequential()
classifier.add(
    Dense(6, input_dim=12, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(Xtrain, yTrain, batch_size=15, epochs=150)

yPred = classifier.predict(Xtest)
yPred = (yPred > 0.5)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3
