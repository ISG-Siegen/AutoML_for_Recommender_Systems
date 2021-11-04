from tpot import TPOTRegressor
from benchmark_framework.model_base import Model


class TPOTHandler(Model):

    def __init__(self):
        super().__init__("TPOT_Regressor", TPOTRegressor(max_time_mins=60), "AutoML")

    def train(self, dataset):
        x_train, y_train = dataset.train_data
        self.model_object.fit(x_train, y_train.values.ravel())

    def predict(self, dataset):
        x_test, _ = dataset.test_data
        return self.model_object.predict(x_test)
