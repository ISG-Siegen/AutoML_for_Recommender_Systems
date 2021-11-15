from benchmark_framework.model_base import Model
from utils.lcer import get_settings_timeoutinmin


def load_h2o_and_all_models():
    import h2o
    from h2o.automl import H2OAutoML

    class H2OHandler(Model):
        def __init__(self):
            h2o.init()

            super().__init__("H2O_H2OGradientBoostingEstimator",
                             H2OAutoML(max_runtime_secs=60 * get_settings_timeoutinmin()), "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data

            features = list(x_train)
            label = list(y_train)

            # Build 1 training frame
            p_x_train = x_train.copy()
            p_x_train[label] = y_train
            # Split
            df = h2o.H2OFrame(p_x_train)
            train, valid = df.split_frame(ratios=[.75])

            self.model_object.train(x=features, y=label[0], training_frame=train, validation_frame=valid)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            df = h2o.H2OFrame(x_test)
            return self.model_object.predict(df).as_data_frame()

    return [H2OHandler]
