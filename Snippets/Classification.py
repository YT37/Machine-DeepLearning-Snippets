# Logistic Regression
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""

# Support Vector Classification
"""from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""

# Naive Bayes
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""

# KNearestNeighbours
"""from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""

# Decision Tree Classification
"""from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""

# Random Forest Classification
"""from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=300, criterion="entropy", random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)
"""

#XGBoost
"""from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=10, n_estimators=300, gamma=0.5, random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)"""