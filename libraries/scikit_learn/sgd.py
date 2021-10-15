from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time


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






