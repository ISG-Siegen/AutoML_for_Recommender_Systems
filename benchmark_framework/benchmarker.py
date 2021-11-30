# General Benchmark class
import time
from benchmark_framework import metrics, dataset_base, model_base
from utils.lcer import get_logger
import traceback

logger = get_logger("Benchmarker")


class Benchmark:

    def __init__(self, dataset: dataset_base.Dataset, metric: metrics.Metric, model: model_base.Model):
        self.dataset = dataset
        self.metric = metric

        # Initiate model such that alternative arguments maybe passed
        if hasattr(model, 'requires_dataset'):
            # In this case, the model is an object that requires additional input parameters (i.e., the dataset)
            self.model = model(dataset)
        else:
            self.model = model()

    # ---- Actual Code
    def _pre(self):
        logger.info(
            "#### Starting benchmark on dataset {} with model {} ####".format(self.dataset.name, self.model.name))

    def _fit_predict(self):
        # Train
        logger.info("# Train " + str(self.model.name) + " #")
        self.model.train(self.dataset)
        # Predict
        logger.info("# Predict " + str(self.model.name) + " #")
        return self.model.predict(self.dataset)

    def run(self):
        self._pre()
        logger.info("### Start Fit and Prediction ###")
        y_pred = self._fit_predict()
        metric_val = self.metric.evaluate(self.dataset, y_pred)

        return metric_val

    def _run_q(self, queue):

        try:
            self._pre()
            logger.info("### Start Fit and Prediction with limits ###")
            y_pred = self._fit_predict()
            metric_val = self.metric.evaluate(self.dataset, y_pred)

            queue.put(metric_val)
        except MemoryError:  # catches numpy memory error for too big allocations
            logger.warning(traceback.format_exc())
            raise SystemExit(9)

    def run_with_limits(self, time_in_min):
        # Function to run relevant code with timeout and handle memory or other errors
        from multiprocessing import Process, Queue

        queue = Queue()

        st = time.time()

        p = Process(target=self._run_q, args=(queue,))

        # Start Process and block for seconds equal to timeout if not returned earlier
        p.start()
        p.join(timeout=int(time_in_min * 60))

        # Vars
        failed = False
        fail_reason = None

        if p.is_alive():
            # Handle correct termination
            p.terminate()
            p.join()

            # Timeout
            failed = True
            fail_reason = "timeout"
        else:
            if p.exitcode == 1:
                # Code ran into a bug -> raise exit
                raise SystemExit(1)
            elif p.exitcode == -9:
                # Algorithm killed because uses too much memory

                failed = True
                fail_reason = "memout"

            elif p.exitcode == 9:  # exit code set by us above
                # Algorithm tried to allocate to much memory
                failed = True
                fail_reason = "memout_allocation"

        # Catch none failed run to get results
        if not failed:
            metric_val = queue.get()
        else:
            metric_val = None

        time_taken = time.time() - st
        return metric_val, time_taken, failed, fail_reason
