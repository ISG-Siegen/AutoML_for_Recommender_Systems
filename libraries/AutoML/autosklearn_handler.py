from benchmark_framework.model_base import Model
from utils.lcer import get_timeout_in_min


def load_auto_sklearn_and_all_models():
    import autosklearn.regression

    class AutoSKLearn(Model):

        def __init__(self):
            super().__init__("AutoSKLearn_AutoSklearnRegressor",
                             autosklearn.regression.AutoSklearnRegressor(
                                 time_left_for_this_task=get_timeout_in_min() * 60),
                             "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [AutoSKLearn]
