from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
from benchmark_framework.model_base import Model


#___________________________________________________MODEL_______________________________________________________________
def sgd(x_train, x_test, y_train, y_test):
    model = make_pipeline(StandardScaler(), SGDRegressor())
    start = time.time()
    model.fit(x_train, y_train.values.ravel())
    stop = time.time()
    predictions = model.predict(x_test)
    execution_time = stop-start
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(predictions)
    return rmse, execution_time


# Model Example for Benchmark Project Structure
class SGD(Model):

    def __init__(self):
        super().__init__("StochasticGradientDescentRegressor", SGDRegressor())
        # FIXME atm without pipeline/standard scaler as it count as preprocessing?

    def train(self, x_train, y_train):
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, x_test):
        return self.model_object.predict(x_test)





