import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from benchmark_framework.model_base import Model


class H2OHandler(Model):
    def __init__(self):
        h2o.init()

        super().__init__("H2O", H2OGradientBoostingEstimator(), "AutoML")

    def train(self, x_train, y_train):
        features = list(x_train)
        label = list(y_train)

        # Build 1 training frame
        p_x_train = x_train.copy()
        p_x_train[label] = y_train
        # Split
        df = h2o.H2OFrame(p_x_train)
        train, valid = df.split_frame(ratios=[.75])

        self.model_object.train(x=features, y=label[0], training_frame=train, validation_frame=valid)

    def predict(self, x_test):
        df = h2o.H2OFrame(x_test)
        return self.model_object.predict(df).as_data_frame()
