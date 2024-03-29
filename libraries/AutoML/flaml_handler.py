from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_flaml_and_all_models():
    from flaml import AutoML

    class FLAMLHandler(Model):

        def __init__(self):
            super().__init__("FLAML_Regressor", AutoML(),
                             "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel(), task="regression", verbose=0,
                                  time_budget=get_timeout_in_min() * 60,
                                  n_jobs=-1,
                                  mem_thres=float("inf"),
                                  metric="rmse"
                                  )

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [FLAMLHandler]
