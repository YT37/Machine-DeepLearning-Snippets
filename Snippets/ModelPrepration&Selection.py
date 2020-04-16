# Reading Data
"""import pandas as pd
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values"""

# Spliting Data
"""from sklearn.model_selection import train_test_split
Xtrain, Xtest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=0)"""

# Scaling Data
"""from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
SCy = StandardScaler()
yTrain = SCy.fit_transform(yTrain.reshape(-1,1))"""

# Missing Data
"""from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])"""

# Cataegorical Data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
y = LabelEncoder().fit_transform(y)

X = X[:, 1:]"""

# Grid Search
"""from sklearn.model_selection import GridSearchCV
params = [{"C":[1, 10, 100, 10000], "kernel":["linear"]}, {"C":[1, 10, 100, 10000], "kernel":["rbf"], "gamma":[0.1, 0.2, 0.3, 0.4, 0.5]},  {"C":[1, 10, 100, 10000], "kernel":["poly"], "degree":[1,2,3,4,5], "gamma":[0.5, 0.1, 0.01, 0.001, 0.0001]}]
gs = GridSearchCV(estimator=classifier, param_grid=params, scoring="accuracy", cv=10, n_jobs=-1)
gs = gs.fit(Xtrain,yTrain)
bestAccu = gs.best_score_
bestParams = gs.best_params_"""

# KFold CV
"""from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()"""

# Kernel PCA
"""from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel="rbf")
Xtrain = kpca.fit_transform(Xtrain)
Xtest = kpca.transform(Xtest)"""

# PCA
"""from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
expVari = pca.explained_variance_ratio_
"""

# LDA
"""from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
Xtrain = lda.fit_transform(Xtrain, yTrain)
Xtest = lda.transform(Xtest)
"""