from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from benchmark_framework.model_base import Model
from abc import abstractmethod


# Default interface for sklearn
class ScikitModel(Model):

    @abstractmethod
    def __init__(self, *args):
        super().__init__(*args)

    def train(self, x_train, y_train):
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, x_test):
        return self.model_object.predict(x_test)


# Different Models from the library
class SVRegressor(ScikitModel):

    def __init__(self):
        super().__init__("SupportVectorRegression", SVR())


class SGD(ScikitModel):

    def __init__(self):
        super().__init__("StochasticGradientDescentRegressor", SGDRegressor())


class RF(ScikitModel):

    def __init__(self):
        super().__init__("RandomForestRegressor", RandomForestRegressor())


class KNN(ScikitModel):

    def __init__(self):
        super().__init__("KNeighborsRegressor", KNeighborsRegressor())
