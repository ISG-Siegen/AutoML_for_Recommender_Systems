from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, name, model_object, category):
        self.name = name
        self.model_object = model_object

        if category not in ["ML", "AutoML", "RecSys", "AutoRecSys", "Baseline"]:
            raise ValueError(("Category Input value {} is not one of the allowed categories:" +
                             " ['ML', 'AutoML', 'RecSys', 'AutoRecSys', 'Baseline'").format(category))
        self.library_category = category

    @abstractmethod
    def train(self, dataset):
        raise NotImplementedError("NotImplementedError: Implementation required for train() method")

    @abstractmethod
    def predict(self, dataset):
        raise NotImplementedError("NotImplementedError: Implementation required for predict() method")
