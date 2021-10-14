from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
from benchmark_framework.model_base import Model


# ___________________________________________________MODEL_______________________________________________________________
def random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor()
    start = time.time()
    model.fit(x_train, y_train.values.ravel())
    stop = time.time()
    predictions = model.predict(x_test)
    execution_time = stop - start
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, execution_time


# Model Example for Benchmark Project Structure
class RF(Model):

    def __init__(self):
        super().__init__("RandomForestRegressor", RandomForestRegressor(n_estimators=2))

    def train(self, x_train, y_train):
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, x_test):
        return self.model_object.predict(x_test)
