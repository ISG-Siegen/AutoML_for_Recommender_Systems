import os
import sys

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Rest of imports
from benchmark_framework import benchmarker, metrics, dataset_base
from utils.lcer import get_logger, get_output_result_data, get_base_path
from utils.filer import write_data, append_data
from datetime import date
from data_processing.preprocessing.preprocessing_100k import load_ml_100k
from libraries.name_lib_mapping import NAME_LIB_MAP
import pandas as pd

# ------------- Start Variables
datasets_list = []
result_data = []
logger = get_logger("BenchmarkExe")

# Read Input (the lib name to run)
lib_name = str(sys.argv[1])
# Load algos from lib name
lib_algos = NAME_LIB_MAP[lib_name]()

#  Collect dataset loaders
dataset_load_functions = [load_ml_100k]
nr_datasets = len(dataset_load_functions)

#  File management
output_filepath = os.path.join(get_base_path(), get_output_result_data(), "{}_{}_overall_benchmark_results.csv".format(
    date.today(), lib_name))
if os.path.isfile(output_filepath):
    os.remove(output_filepath)
write_data(pd.DataFrame([], columns=["Dataset", "Model", "LibraryCategory", "RSME", "TimeInSeconds"]), output_filepath)

# ------------- Loop over all datasets
logger.info("######## Loop over all Datasets and do benchmarks for library {} ########".format(lib_name))
for idx, dataset_function in enumerate(dataset_load_functions, 1):
    # Load dataset and create dataset object
    logger.info("###### Load Datasets {}/{} ######".format(idx, nr_datasets))
    dataset = dataset_base.Dataset(*load_ml_100k())
    logger.info("###### Start processing Dataset {} ######".format(dataset.name))

    # Build benchmark for this dataset
    benchmarks = []
    for model_base in lib_algos:
        benchmarks.append(benchmarker.Benchmark(dataset, metrics.RSME(), 60, model_base()))

    # Execute benchmarks for this dataset
    for benchmark in benchmarks:
        tmp_result_data = [(dataset.name, benchmark.model.name, benchmark.model.library_category, *benchmark.run())]
        logger.info("###### Intermediate Result Output and Saving Data ######")
        _, model_name, _, metric_val, execution_time = tmp_result_data[0]
        logger.info("{}: RSME of {} | Time take {}".format(model_name, metric_val, execution_time))

        # Build tmp df to output data
        append_data(
            pd.DataFrame(tmp_result_data, columns=["Dataset", "Model", "LibraryCategory", "RSME", "TimeInSeconds"]),
            output_filepath)

    logger.info("###### Finished processing Dataset {} ######".format(dataset.name))

logger.info("######## Benchmark finished ########")
