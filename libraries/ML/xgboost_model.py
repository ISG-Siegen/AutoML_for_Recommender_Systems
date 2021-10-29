from benchmark_framework.model_base import Model


def load_xgboost_and_all_models():
    from xgboost import XGBRegressor

    class XGBoostModel(Model):

        def __init__(self):
            super().__init__("XGBoostRegressor", XGBRegressor(), "ML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            self.model_object.fit(x_train, y_train.values.ravel())

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [XGBoostModel]
