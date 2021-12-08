from benchmark_framework.model_base import Model
from general_utils.lcer import get_timeout_in_min


def load_h2o_and_all_models():
    import h2o
    from h2o.automl import H2OAutoML
    import psutil
    import logging
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    class H2OHandler(Model):
        def __init__(self):
            # Following Memory Usage from:
            # https://github.com/openml/automlbenchmark/blob/master/frameworks/H2OAutoML/exec.py
            jvm_memory = str(round((psutil.virtual_memory().total / 1e+6) * 2 / 3)) + "M"

            h2o.init(nthreads=-1,
                     max_mem_size=jvm_memory,  # Only supports mem limit like this, will take 25% of memory by default
                     min_mem_size=jvm_memory
                     )

            super().__init__("H2O_AutoML",
                             H2OAutoML(max_runtime_secs=60 * get_timeout_in_min(),
                                       sort_metric="RMSE"
                                       ), "AutoML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data

            features = list(x_train)
            label = list(y_train)

            # Build 1 training frame
            p_x_train = x_train.copy()
            p_x_train[label] = y_train
            # Split
            train = h2o.H2OFrame(p_x_train)

            self.model_object.train(x=features, y=label[0], training_frame=train)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            df = h2o.H2OFrame(x_test)
            return self.model_object.predict(df).as_data_frame()

    return [H2OHandler]
