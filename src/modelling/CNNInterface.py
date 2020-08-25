# We will define an interface that will expose the
# main methods that the process of training/testing for
# a convolutional neural network must contain

class CNNInterface:
    def train(self, X_train, Y_train, X_test, Y_test):
        pass

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        pass

    def save_model(self):
        pass

    def confusion_matrix(self, X_test, Y_test_values):
        pass
