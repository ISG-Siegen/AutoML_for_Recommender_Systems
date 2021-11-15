from benchmark_framework.model_base import Model
from utils.lcer import get_timeout_in_min


def load_gama_and_all_models():
    from gama import GamaRegressor

    class GamaHandler(Model):

        def __init__(self):
            super().__init__("GAMA_Regressor", GamaRegressor(max_total_time=get_timeout_in_min() * 60,
                                                             store="nothing"),  # nothing to reduce logs
                             "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [GamaHandler]
