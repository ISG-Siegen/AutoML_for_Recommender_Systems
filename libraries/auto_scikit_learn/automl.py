from sklearn.metrics import mean_squared_error
import autosklearn.regression
import time
from benchmark_framework.model_base import Model

# ___________________________________________________MODEL_______________________________________________________________
def automl(x_train, x_test, y_train, y_test):
    model = autosklearn.regression.AutoSklearnRegressor()
    start = time.time()
    model.fit(x_train, y_train.values.ravel())
    stop = time.time()
    predictions = model.predict(x_test)
    execution_time = stop - start
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(predictions)
    return rmse, execution_time


class AutoSKLearn(Model):

    def __init__(self):
        super().__init__("AutoSKLearn", autosklearn.regression.AutoSklearnRegressor(), "AutoML")

    def train(self, x_train, y_train):
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, x_test):
        return self.model_object.predict(x_test)
