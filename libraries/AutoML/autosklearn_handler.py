import autosklearn.regression
from benchmark_framework.model_base import Model


class AutoSKLearn(Model):

    def __init__(self):
        super().__init__("AutoSKLearn", autosklearn.regression.AutoSklearnRegressor(),
                         "AutoML")

    def train(self, x_train, y_train):
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, x_test):
        return self.model_object.predict(x_test)
