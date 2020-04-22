# ANN

"""from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

classifier = Sequential()
classifier.add(
    Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer="rmsprop",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
classifier.fit(Xtrain, yTrain, batch_size=25, epochs=500)

yPred = classifier.predict(Xtest)"""