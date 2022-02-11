from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_auto_sklearn_and_all_models():
    import autosklearn.regression
    from autosklearn.metrics import root_mean_squared_error
    import math
    import multiprocessing
    import psutil

    # To set memory limit for autosklearn
    # we follow the automl benchmark code https://github.com/automl/auto-sklearn/blob/master/autosklearn/estimators.py
    vm = psutil.virtual_memory()
    total_memory_mb = vm.total / (1 << 20)
    max_mem_size_mb = vm.available / (1 << 20)
    n_jobs = multiprocessing.cpu_count()
    ml_memory_limit = max(min(max_mem_size_mb / n_jobs, math.ceil(total_memory_mb / n_jobs)), 3072)

    class AutoSKLearn(Model):

        def __init__(self):
            super().__init__("AutoSKLearn_AutoSklearnRegressor",
                             autosklearn.regression.AutoSklearnRegressor(
                                 time_left_for_this_task=get_timeout_in_min() * 60,
                                 memory_limit=ml_memory_limit,
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
