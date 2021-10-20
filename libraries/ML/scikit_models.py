from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from benchmark_framework.model_base import Model
from abc import abstractmethod


# Default interface for sklearn
class ScikitModel(Model):

    @abstractmethod
    def __init__(self, name, model):
        super().__init__("Scikit_" + name, model, "ML")

    def train(self, dataset):
        x_train, y_train = dataset.train_data
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, dataset):
        x_test, _ = dataset.test_data
        return self.model_object.predict(x_test)


# Different Models from the library
class ScikitSVRegressor(ScikitModel):

    def __init__(self):
        super().__init__("SciKit_SupportVectorRegression", SVR())


class ScikitSGD(ScikitModel):

    def __init__(self):
        super().__init__("SciKit_StochasticGradientDescentRegressor", SGDRegressor())


class ScikitRF(ScikitModel):

    def __init__(self):
        super().__init__("SciKit_RandomForestRegressor", RandomForestRegressor())


class ScikitKNN(ScikitModel):

    def __init__(self):
        super().__init__("SciKit_KNeighborsRegressor", KNeighborsRegressor())
