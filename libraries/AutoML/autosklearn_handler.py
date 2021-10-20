import autosklearn.regression
from benchmark_framework.model_base import Model


class AutoSKLearn(Model):

    def __init__(self):
        super().__init__("AutoSKLearn_AutoSklearnRegressor", autosklearn.regression.AutoSklearnRegressor(),
                         "AutoML")

    def train(self, dataset):
        x_train, y_train = dataset.train_data
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, dataset):
        x_test, _ = dataset.test_data
        return self.model_object.predict(x_test)
