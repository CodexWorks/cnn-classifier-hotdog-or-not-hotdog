import keras
import matplotlib.pyplot as plt
import numpy as np

from src.modelling.CNNInterface import CNNInterface
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, \
    BatchNormalization, Flatten
from sklearn.metrics import confusion_matrix
from src.utilities.constants import dataset_path_models


class MainCNNRecognition(CNNInterface):
    def __init__(self, name, input_shape):
        self.__input_shape = input_shape
        self.__name = name
        self.__score_train = []
        self.__score_test = []
        self.__history = None
        # self.__augmentationImageGenerator = augmentationImageGenerator

        print("Initializing CNN")

        self.__model = Sequential()
        self.__model.add(Conv2D(64, kernel_size=(3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                input_shape=input_shape))
        self.__model.add(MaxPooling2D((2, 2)))
        self.__model.add(BatchNormalization())
        self.__model.add(Dropout(0.25))

        self.__model.add(Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Dropout(0.25))

        self.__model.add(Conv2D(128, (3, 3), activation='relu'))
        self.__model.add(Dropout(0.3))

        self.__model.add(Flatten())
        self.__model.add(Dense(128, activation='relu'))

        self.__model.add(Dropout(0.3))
        self.__model.add(Dense(2, activation='softmax'))

        print("CNN Initialized")

    def __str__(self):
        return str(self.__model.summary())

    def train(self, X_train, Y_train, X_test, Y_test):
        print("Start training models")

        self.__model.compile(loss="binary_crossentropy",
                             optimizer=keras.optimizers.Adam(lr=0.0001),
                             metrics=['accuracy'])

        # iterator = self.__augmentationImageGenerator.flow(X_train, Y_train)

        self.__history = self.__model.fit(
            X_train, Y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, Y_test))

        print("Training completed")

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        self.__score_train = self.__model.evaluate(
            x=X_train,
            y=Y_train)

        self.__score_test = self.__model.evaluate(
            x=X_test,
            y=Y_test)

        print("Train loss", self.__score_train[0])
        print("Train accuracy", self.__score_train[1])

        print("Test loss", self.__score_test[0])
        print("Test accuracy", self.__score_test[1])

        return [self.__score_train, self.__score_test]

    def save_model(self):
        try:
            model_path = dataset_path_models + self.__name + ".h5"
            self.__model.save(model_path)
        except:
            print("An exception occurred")

    def confusion_matrix(self, X_test, Y_test_values):
        # Predict
        y_pred = self.__model.predict_classes(X_test)
        Y_test_values = np.argmax(Y_test_values, axis=1)
        # Generate confusion matrix
        conf_matrix = confusion_matrix(Y_test_values, y_pred)
        print(conf_matrix)

    def plot_history(self):
        if self.__history is not None:
            # Summarize history for accuracy
            plt.plot(self.__history.history['accuracy'])
            plt.plot(self.__history.history['val_accuracy'])
            plt.title('models accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            # Summarize history for loss
            plt.plot(self.__history.history['loss'])
            plt.plot(self.__history.history['val_loss'])
            plt.title('models loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
