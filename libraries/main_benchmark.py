import os
import sys

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Rest of imports
from benchmark_framework import benchmarker, metrics, dataset_base
from utils.lcer import get_logger, get_output_result_data, get_base_path
from utils.filer import write_data, append_data
import time
from libraries.name_lib_mapping import NAME_LIB_MAP
import pandas as pd
from data_processing.preprocessing.main_preprocessing import get_dataset_load_functions

# ------------- Start Variables
datasets_list = []
result_data = []
logger = get_logger("BenchmarkExe")

# Read Input (the lib name to run)
lib_name = str(sys.argv[1])
fresh_start = False

# Load algos from lib name
lib_algos = NAME_LIB_MAP[lib_name]()

#  Collect dataset loaders
dataset_load_functions = get_dataset_load_functions()[:1]
nr_datasets = len(dataset_load_functions)

# ------------- File management
output_filepath = os.path.join(get_base_path(), get_output_result_data(), "overall_benchmark_results.csv")

if fresh_start and os.path.isfile(output_filepath):
    os.remove(output_filepath)

if not os.path.isfile(output_filepath) or fresh_start:
    write_data(pd.DataFrame([], columns=["timestamp", "Dataset", "Model", "LibraryCategory", "RSME", "TimeInSeconds"]),
               output_filepath)

# ------------- Loop over all datasets
logger.info("######## Loop over all Datasets and do benchmarks for library {} ########".format(lib_name))
for idx, dataset_load_function in enumerate(dataset_load_functions, 1):
    # Load dataset and create dataset object
    logger.info("###### Load Datasets {}, {}/{} ######".format(dataset_load_function, idx, nr_datasets))
    dataset = dataset_base.Dataset(*dataset_load_function())
    logger.info("###### Start processing Dataset {} ######".format(dataset.name))

    # Build metric once and not in every loop
    metric = metrics.RSME()

    # Execute benchmarks for every algorithm
    for model_base in lib_algos:
        #  Build benchmark for this dataset and algorithm
        benchmark = benchmarker.Benchmark(dataset, metric, 60, model_base)
        # Execute benchmarks for this dataset
        tmp_result_data = [(time.time(), dataset.name, benchmark.model.name, benchmark.model.library_category,
                            *benchmark.run())]

        logger.info("###### Intermediate Result Output and Saving Data ######")
        _, _, model_name, _, metric_val, execution_time = tmp_result_data[0]
        logger.info("{}: RSME of {} | Time taken {}".format(model_name, metric_val, execution_time))

        # Build tmp df to output data
        append_data(pd.DataFrame(tmp_result_data, columns=["datetime", "Dataset", "Model", "LibraryCategory", "RSME",
                                                           "TimeInSeconds"]), output_filepath)

    logger.info("###### Finished processing Dataset {} ######".format(dataset.name))

logger.info("######## Benchmark finished ########")
