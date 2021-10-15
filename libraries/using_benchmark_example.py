import os
from benchmark_framework import benchmarker, metrics, dataset_base
from libraries.scikit_learn.scikit_models import RF, SVRegressor, SGD, KNN
from data_processing.preprocessing.preprocessing_100k import load_ml_100k
from libraries.H2O.H2O import H2OHandler
from libraries.auto_scikit_learn.automl import AutoSKLearn
import pandas as pd
from utils.lcer import get_logger, get_output_result_data
from utils.filer import write_data

logger = get_logger("BenchmarkExe")

# ------------- Start Variables
datasets_list = []
result_data = []

# ------------- Load and collect datasets
logger.info("######## Load Datasets ########")
data, features, label = load_ml_100k()
datasets_list.append(dataset_base.Dataset("movielens-100k", data, features, label))

# ------------- Loop over all datasets
logger.info("######## Loop over all Datasets and do benchmarks ########")
for dataset in datasets_list:
    # Build benchmark for this dataset
    benchmarks = []
    for model_base in [KNN, RF, SGD, SVRegressor, H2OHandler, AutoSKLearn]:
        benchmarks.append(benchmarker.Benchmark(dataset, metrics.RSME(), 60, model_base()))

    # Execute benchmarks for this dataset
    tmp_result_data = []
    for benchmark in benchmarks:
        tmp_result_data.append((dataset.name, benchmark.model.name, benchmark.model.library_category, *benchmark.run()))

    # Evaluate Intermediate results for this dataset
    for _, model_name, _, metric_val, execution_time in tmp_result_data:
        print("{}: RSME of {} | Time take {}".format(model_name, metric_val, execution_time))

    # Add result of this dataset to full collection
    result_data.extend(tmp_result_data)


# ------------- Output Data as results file
logger.info("######## Export Result data ########")
out_df = pd.DataFrame(result_data, columns=["Dataset", "Model", "LibraryCategory", "RSME", "TimeInSeconds"])
write_data(out_df, os.path.join("." + get_output_result_data(), "overall_benchmark_results.csv"))
