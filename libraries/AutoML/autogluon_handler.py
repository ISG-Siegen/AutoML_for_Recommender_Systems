from benchmark_framework.model_base import Model
from utils.lcer import get_timeout_in_min


def load_autogluon_and_all_models():
    from autogluon.tabular import TabularPredictor
    
    class AutoGluonHandler(Model):

        def __init__(self):
            super().__init__("AutoGluon_Regressor", None, "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data

            # Build dataset
            merged_train_data = x_train.copy()
            merged_train_data[dataset.label] = y_train

            # Build Model and fit
            # verbosity sets our own output to verbosity = 0, hence have to keep verbosity=2 (i.e. the default value)
            self.model_object = TabularPredictor(label=dataset.label, eval_metric="root_mean_squared_error")
            # not able to set memory limit or jobs count, takes by default all available
            self.model_object.fit(merged_train_data, time_limit=60 * get_timeout_in_min())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [AutoGluonHandler]
