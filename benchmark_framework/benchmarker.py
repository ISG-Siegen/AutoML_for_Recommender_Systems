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

        # Initiate model such that alternative arguments maybe passed
        if hasattr(model, 'requires_dataset'):
            # In this case, the model is an object that requires additional input parameters (i.e., the dataset)
            self.model = model(dataset)
        else:
            self.model = model()

    def run(self):
        logger.info(
            "#### Starting benchmark on dataset {} with model {} ####".format(self.dataset.name, self.model.name))

        # Run dataset while measuring and being in budget
        logger.info("### Start Training and Prediction with timing ###")
        start_time = time.time()

        # TODO add budget wrapper around the code here
        # Hard budget via subprocess, or
        # Soft budget via library itself

        # Train
        logger.info("# Train " + str(self.model.name) + " #")  # FIXME For true time benchmark remove this logger?
        self.model.train(self.dataset)
        # Predict
        logger.info("# Predict " + str(self.model.name) + " #")  # FIXME For true time benchmark remove this logger?
        y_pred = self.model.predict(self.dataset)

        # Get time
        execution_time = time.time() - start_time

        # Calc metric
        metric_val = self.metric.evaluate(self.dataset, y_pred)

        # Return metric and time
        return metric_val, execution_time
