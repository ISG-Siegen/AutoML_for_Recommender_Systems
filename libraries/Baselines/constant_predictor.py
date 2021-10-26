import pandas as pd

from benchmark_framework.model_base import Model


class MeanPredictor(Model):

    def __init__(self):
        super().__init__("ConstantPredictor_Mean", 0,
                         "Baseline")

    def train(self, dataset):
        x_train, y_train = dataset.train_data
        self.model_object = y_train.mean()

    def predict(self, dataset):
        x_test, _ = dataset.test_data
        return [self.model_object]*x_test.shape[0]