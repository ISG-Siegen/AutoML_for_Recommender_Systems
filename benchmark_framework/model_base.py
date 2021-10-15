from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, name, model_object):
        self.name = name
        self.model_object = model_object

    @abstractmethod
    def train(self, x_train, y_train):
        raise NotImplementedError("NotImplementedError: Implementation required for train() method")

    @abstractmethod
    def predict(self, x_test):
        raise NotImplementedError("NotImplementedError: Implementation required for predict() method")
