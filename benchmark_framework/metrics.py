from sklearn.metrics import mean_squared_error
from abc import abstractmethod


class Metric:
    @abstractmethod
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, dataset, y_pred):
        pass


class RSME(Metric):

    def __init__(self):
        super().__init__("RSME")

    def evaluate(self, dataset, y_pred):
        _, y_true = dataset.test_data
        return mean_squared_error(y_true, y_pred, squared=False)
