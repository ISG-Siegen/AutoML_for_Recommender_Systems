from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_auto_pytorch_and_all_models():
    from autoPyTorch.api.tabular_regression import TabularRegressionTask
    import math
    import multiprocessing
    import psutil

    # To set memory limit for autopytorch we use autosklearn code (as it has the same memory limit description/api)
    # we follow the automl benchmark code https://github.com/automl/auto-sklearn/blob/master/autosklearn/estimators.py
    vm = psutil.virtual_memory()
    total_memory_mb = vm.total / (1 << 20)
    max_mem_size_mb = vm.available / (1 << 20)
    n_jobs = multiprocessing.cpu_count()
    ml_memory_limit = max(min(max_mem_size_mb / n_jobs, math.ceil(total_memory_mb / n_jobs)), 3072)

    class AutoPyTorch(Model):

        def __init__(self):
            super().__init__("AutoPytorch_TabularRegressor",
                             TabularRegressionTask(
                                 n_jobs=multiprocessing.cpu_count(),
                                 delete_tmp_folder_after_terminate=True,
                             ),
                             "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            # Make sure y_train is float type and add small value too one of the values
            # to make sure autopytorch does not think this is a classification task
            y_train = y_train.values.ravel().astype(float)
            y_train[-1] = y_train[-1] + 0.00000001

            self.model_object.search(X_train=x_train, y_train=y_train,
                                     total_walltime_limit=get_timeout_in_min() * 60,
                                     memory_limit=ml_memory_limit,
                                     optimize_metric="root_mean_squared_error",
                                     budget_type="runtime",
                                     )

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [AutoPyTorch]
