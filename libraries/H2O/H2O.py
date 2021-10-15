import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import time
from benchmark_framework.model_base import Model


# ___________________________________________________MODEL_______________________________________________________________
def H2O(pandas_df):
    h2o.init()
    dataFrame = h2o.H2OFrame(pandas_df)
    train, valid = dataFrame.split_frame(ratios=[.75])
    model = H2OGradientBoostingEstimator()
    start = time.time()
    model.train(x=["movieId", "userId", 'unknown', 'action',
                   'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                   'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                   'war', 'western'], y="rating", training_frame=train, validation_frame=valid)
    stop = time.time()
    execution_time = stop - start
    rmse = model.rmse()
    return rmse, execution_time


class H2OHandler(Model):
    def __init__(self):
        h2o.init()

        super().__init__("H2O", H2OGradientBoostingEstimator())

    def train(self, x_train, y_train):
        features = list(X_train)
        label = list(y_train)

        # Build 1 training frame
        x_train[label] = y_train
        # Split
        train, valid = dataFrame.split_frame(ratios=[.75])

        self.model_object.train(x=features, y=label[0], training_frame=train, validation_frame=valid)

    def predict(self, x_test):
        return self.model_object.predict(x_test)
