from sklearn.metrics import mean_squared_error
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


class RSME(Metric):

    def __init__(self):
        super().__init__("RSME")

    def evaluate(self, dataset, y_pred):
        _, y_true = dataset.test_data

        # TODO unify this or remove dependence on sklearn + packaging in general
        sklearn_version = version.parse(__version__)
        if sklearn_version >= version.parse("0.22"):
            return mean_squared_error(y_true, y_pred, squared=False)
        elif sklearn_version < version.parse("0.22"):
            from math import sqrt
            return sqrt(mean_squared_error(y_true, y_pred))
        else:
            raise RuntimeError("Unknown sklearn version {}".format(__version__))


# Constant for string to metric mapping
NAME_TO_METRIC = {
    "RSME": RSME
}
