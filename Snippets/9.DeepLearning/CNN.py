from tensorflow.keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

trainData = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

testData = ImageDataGenerator(rescale=1./255)

trainSet = trainData.flow_from_directory(
        directory=r"../1.Datasets/CNN/train",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testSet = testData.flow_from_directory(
        directory=r"../1.Datasets/CNN/test",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainSet,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testSet,
        validation_steps=2000)
