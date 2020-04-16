# KMeans Clustering
"""from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number Of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init="k-means++",
                max_iter=300, n_init=10, random_state=0)
yKmeans = kmeans.fit_predict(X)"""

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