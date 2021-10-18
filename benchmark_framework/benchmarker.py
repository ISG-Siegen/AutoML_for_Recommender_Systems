# General Benchmark class
import time
from benchmark_framework import metrics, dataset_base, model_base
from utils.lcer import get_logger

logger = get_logger("Benchmarker")


class Benchmark:

    def __init__(self, dataset: dataset_base.Dataset, metric: metrics.Metric, budget, model: model_base.Model):
        self.dataset = dataset
        self.metric = metric
        self.budget = budget
        self.model = model

    def run(self):
        logger.info(
            "###### Starting benchmark on dataset {} with model {} ######".format(self.dataset.name, self.model.name))

        # Get required data
        x_train, y_train = self.dataset.train_data
        x_test, y_test = self.dataset.test_data

        # Run dataset while measuring and being in budget
        logger.info("### Start Training and Prediction with timing ###")
        start_time = time.time()

        # TODO add budget wrapper around the code here
        # Hard budget via subprocess, or
        # Soft budget via library itself

        # Train
        logger.info("# Train #")  # FIXME For true time benchmark remove this logger?
        self.model.train(x_train, y_train)
        # Predict
        logger.info("# Predict #")  # FIXME For true time benchmark remove this logger?
        y_pred = self.model.predict(x_test)

        # Get time
        execution_time = time.time() - start_time

        # Calc metric
        logger.info("### Calculate benchmark results ###")
        metric_val = self.metric.evaluate(y_test, y_pred)

        # Return metric and time
        logger.info("### Finished and return ###")
        return metric_val, execution_time
