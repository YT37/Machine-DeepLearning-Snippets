import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv(r"../1.Datasets/Wine.csv")

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

Xtrain, Xtest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=0)

scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)

pca = PCA(n_components=2)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
expVari = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3

Xset, ySet = Xtrain, yTrain
X1, X2 = np.meshgrid(np.arange(start=Xset[:, 0].min() - 1, stop=Xset[:, 0].max() + 1, step=0.01),
                     np.arange(start=Xset[:, 1].min() - 1, stop=Xset[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', "blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(Xset[ySet == j, 0], Xset[ySet == j, 1],
                c=ListedColormap(('red', 'green', "blue"))(i), label=j)
plt.title('Logistic Regression PCA (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

Xset, ySet = Xtest, yTest
X1, X2 = np.meshgrid(np.arange(start=Xset[:, 0].min() - 1, stop=Xset[:, 0].max() + 1, step=0.01),
                     np.arange(start=Xset[:, 1].min() - 1, stop=Xset[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', "blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(Xset[ySet == j, 0], Xset[ySet == j, 1],
                c=ListedColormap(('red', 'green', "blue"))(i), label=j)
plt.title('Logistic Regression PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
