from benchmark_framework.model_base import Model
from utils.lcer import get_timeout_in_min


def load_tpot_and_all_models():
    from tpot import TPOTRegressor

    class TPOTHandler(Model):

        def __init__(self):
            super().__init__("TPOT_Regressor", TPOTRegressor(generations=None,
                                                             max_time_mins=get_timeout_in_min()), "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [TPOTHandler]
