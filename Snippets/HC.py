# Hierarchial Clustering
"""import scipy.cluster.hierarchy as hc
dendogram = hc.dendrogram(hc.linkage(X,method="ward"))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Eucleidian Distances")
plt.show

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(
    n_clusters=5, affinity="euclidean", linkage="ward")
yHc = hc.fit_predict(X)"""