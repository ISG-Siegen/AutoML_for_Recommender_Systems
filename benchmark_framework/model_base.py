from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, name, model_object, category):
        self.name = name
        self.model_object = model_object

        if category not in ["ML", "AutoML", "RecSys", "AutoRecSys"]:
            raise ValueError(("Category Input value {} is not one of the allowed categories:" +
                             " ['ML', 'AutoML', 'RecSys', 'AutoRecSys'").format(category))
        self.library_category = category

    @abstractmethod
    def train(self, x_train, y_train):
        raise NotImplementedError("NotImplementedError: Implementation required for train() method")

    @abstractmethod
    def predict(self, x_test):
        raise NotImplementedError("NotImplementedError: Implementation required for predict() method")
