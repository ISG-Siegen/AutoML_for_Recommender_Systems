from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_auto_sklearn_and_all_models():
    import autosklearn.regression
    from autosklearn.metrics import root_mean_squared_error

    class AutoSKLearn(Model):

        def __init__(self):
            super().__init__("AutoSKLearn_AutoSklearnRegressor",
                             autosklearn.regression.AutoSklearnRegressor(
                                 time_left_for_this_task=get_timeout_in_min() * 60,
                                 memory_limit=None,
                                 n_jobs=-1,
                                 metric=root_mean_squared_error
                             ),
                             "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [AutoSKLearn]
