import os
import sys

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Rest of imports
from benchmark_framework import benchmarker, dataset_base
from general_utils.lcer import get_logger, get_output_result_data, get_base_path, get_default_metric, \
    get_hard_timeout_in_min
from general_utils import filer
import time
from libraries.name_lib_mapping import NAME_LIB_MAP
import pandas as pd
from data_processing.preprocessing.main_preprocessing import get_dataset_load_functions, DATASET_NAMES

# Use this to guard against multiprocess error
if __name__ == '__main__':
    # ------------- Start Variables
    datasets_list = []
    result_data = []
    logger = get_logger("BenchmarkExe")

    # Read Input (the lib name to run)
    lib_name = "sklearn"  # str(sys.argv[1])
    only_new_benchmarks = True
    with_limits = True

    # Load algos from lib name
    lib_algos = NAME_LIB_MAP[lib_name]()

    #  Collect dataset loaders
    dataset_load_functions = get_dataset_load_functions()
    nr_datasets = len(dataset_load_functions)

    # Default values
    metric = get_default_metric()()
    hard_timeout = get_hard_timeout_in_min()

    # ------------- File management
    output_filepath = os.path.join(get_base_path(), get_output_result_data(), "overall_benchmark_results.csv")

    if only_new_benchmarks:
        run_so_far = filer.read_data_json(
            os.path.join(get_base_path(), get_output_result_data(), "run_overhead_data.json"))
        # Only select subset of data relevant to the current run
        run_so_far_lib = run_so_far[lib_name]

        if run_so_far_lib["all_benchmarks_done_at_least_once"]:
            logger.info("### Library already fully benchmarked - Exit Script ###")
            exit(0)

    # ------------- Loop over all datasets
    logger.info("######## Loop over all Datasets and do benchmarks for library {} ########".format(lib_name))
    for idx, dataset_load_function in enumerate(dataset_load_functions, 1):

        # Check if dataset can be skipped
        if only_new_benchmarks and (not run_so_far_lib[DATASET_NAMES[idx - 1]]):
            # Skip
            logger.info(
                "###### Skip dataset {} as all models for it have been run before ######".format(
                    DATASET_NAMES[idx - 1]))
            continue

        # Load dataset and create dataset object
        logger.info("###### Load Datasets {}, {}/{} ######".format(dataset_load_function, idx, nr_datasets))
        dataset = dataset_base.Dataset(*dataset_load_function())
        logger.info("###### Start processing Dataset {} ######".format(dataset.name))

        # Execute benchmarks for every algorithm
        for model_base in lib_algos:
            #  Build benchmark for this dataset and algorithm
            benchmark = benchmarker.Benchmark(dataset, metric, model_base)

            # Check if the model has to be run at all
            if only_new_benchmarks and benchmark.model.name not in run_so_far_lib[dataset.name]:
                # Skip
                logger.info("###### Skip {} as it has been run before ######".format(benchmark.model.name))
                continue

            # Run benchmark
            if with_limits:
                # Execute benchmark with limits
                metric_val, time_taken, failed, fail_reason = benchmark.run_with_limits(hard_timeout)
            else:
                metric_val = benchmark.run()
                time_taken = failed = fail_reason = None

            logger.info("###### Intermediate Result Output and Saving Data ######")
            logger.info("{}: {} of {} | Time taken: {} | Failed: {} - {}".format(metric.name, benchmark.model.name,
                                                                                 metric_val, time_taken,
                                                                                 failed, fail_reason))

            # Build tmp df to output data
            tmp_result_data = [(dataset.name, benchmark.model.name, benchmark.model.library_category,
                                metric_val, time_taken, time.time(), failed, fail_reason)]
            filer.append_data(pd.DataFrame(tmp_result_data,
                                           columns=["Dataset", "Model", "LibraryCategory", metric.name,
                                                    "TimeInSeconds", "timestamp", "failed", "fail_reason"]),
                              output_filepath)

        logger.info("###### Finished processing Dataset {} ######".format(dataset.name))

    logger.info("######## Benchmark finished ########")
