from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time


#___________________________________________________MODEL_______________________________________________________________
def random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor()
    start = time.time()
    model.fit(x_train, y_train.values.ravel())
    stop = time.time()
    predictions = model.predict(x_test)
    execution_time = stop-start
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, execution_time







