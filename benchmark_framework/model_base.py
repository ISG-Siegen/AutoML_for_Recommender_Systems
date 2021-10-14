from abc import abstractmethod


class Model:
    @abstractmethod
    def __init__(self, name, model_object):
        self.name = name
        self.model_object = model_object

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass
