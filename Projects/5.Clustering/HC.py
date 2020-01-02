import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("MallCustomers.csv")

X = dataset.iloc[:, [3, 4]].values

"""import scipy.cluster.hierarchy as hc
dendogram = hc.dendrogram(hc.linkage(X,method="ward"))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Eucleidian Distances")
plt.show"""

hc = AgglomerativeClustering(
    n_clusters=5, affinity="euclidean", linkage="ward")
yHc = hc.fit_predict(X)


plt.scatter(X[yHc == 0, 0], X[yHc == 0, 1], s=100, c="red", label="Careful")
plt.scatter(X[yHc == 1, 0], X[yHc == 1, 1], s=100, c="blue", label="Standard")
plt.scatter(X[yHc == 2, 0], X[yHc == 2, 1], s=100, c="green", label="Target")
plt.scatter(X[yHc == 3, 0], X[yHc == 3, 1], s=100, c="cyan", label="Careless")
plt.scatter(X[yHc == 4, 0], X[yHc == 4, 1],
            s=100, c="yellow", label="Sensible")
plt.title("Clusters Of Customers")
plt.xlabel("Anual Income (K$)")
plt.ylabel("Spendeing Scoore (1-180)")
plt.legend()
plt.show()
