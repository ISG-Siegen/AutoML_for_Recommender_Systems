from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_auto_pytorch_and_all_models():
    from autoPyTorch.api.tabular_regression import TabularRegressionTask
    import multiprocessing

    class AutoSKLearn(Model):

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
                                     memory_limit=None,
                                     optimize_metric="root_mean_squared_error",
                                     budget_type="runtime",
                                     )

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [AutoSKLearn]
