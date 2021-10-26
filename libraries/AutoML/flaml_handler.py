from flaml import AutoML
from benchmark_framework.model_base import Model


class FLAMLHandler(Model):

    def __init__(self):
        super().__init__("FLAML_Regressor", AutoML(),
                         "AutoML")

    def train(self, dataset):
        x_train, y_train = dataset.train_data
        self.model_object.fit(x_train, y_train.values.ravel(), task="regression")

    def predict(self, dataset):
        x_test, _ = dataset.test_data
        return self.model_object.predict(x_test)
