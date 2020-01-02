from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
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
        directory='dataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testSet = testData.flow_from_directory(
        directory='dataset/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainSet,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testSet,
        validation_steps=2000)

classifier.save("D:\Codes\Machine Learning\Section 40 - Convolutional Neural Networks (CNN)")