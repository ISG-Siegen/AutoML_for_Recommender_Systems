from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import __version__
from abc import abstractmethod
from packaging import version


class Metric:
    @abstractmethod
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, dataset, y_pred):
        pass


class RMSE(Metric):

    def __init__(self):
        super().__init__("RMSE")

    def evaluate(self, dataset, y_pred):
        _, y_true = dataset.test_data

        # TODO unify this or remove dependence on sklearn + packaging in general
        # Depending on the environment of a library, the sklearn version can differ and be too old for the RMSE.
        # The below is a workaround.
        sklearn_version = version.parse(__version__)
        if sklearn_version >= version.parse("0.22"):
            return mean_squared_error(y_true, y_pred, squared=False)
        elif sklearn_version < version.parse("0.22"):
            from math import sqrt
            return sqrt(mean_squared_error(y_true, y_pred))
        else:
            raise RuntimeError("Unknown sklearn version {}".format(__version__))


class MAE(Metric):

    def __init__(self):
        super().__init__("MAE")

    def evaluate(self, dataset, y_pred):
        _, y_true = dataset.test_data
        return mean_absolute_error(y_true, y_pred)


# Constant for string to metric mapping
NAME_TO_METRIC = {
    "RMSE": RMSE,
    "MAE": MAE
}
