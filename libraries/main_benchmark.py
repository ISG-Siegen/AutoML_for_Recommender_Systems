import os
from benchmark_framework import benchmarker, metrics
from data_processing.preprocessing.main_preprocessing import load_file_to_csv
from libraries.ML.scikit_models import load_all_scikit_models
from libraries.AutoML.H2O_handler import H2OHandler
from libraries.RecSys.surprise import load_all_surprise_models
from libraries.AutoML.autosklearn_handler import AutoSKLearn
from libraries.AutoRecSys.auto_surprise import AutoSurpriseModel
import pandas as pd
from utils.lcer import get_logger, get_output_result_data, get_base_path
from utils.filer import write_data
from datetime import date
from libraries.AutoML.tpot_handler import TPOTHandler
from libraries.Baselines.constant_predictor import MeanPredictor
from libraries.ML.xgboosst_model import XGBoostModel
from libraries.AutoML.flaml_handler import FLAMLHandler
from libraries.AutoML.gama_handler import GamaHandler

logger = get_logger("BenchmarkExe")

imported_models = [AutoSurpriseModel, H2OHandler, AutoSKLearn, TPOTHandler, MeanPredictor, XGBoostModel,
                   FLAMLHandler, GamaHandler] + load_all_scikit_models() + load_all_surprise_models()

# ------------- Start Variables
result_data = []

# ------------- Load Datasets
datasets_list = load_file_to_csv(logger)

# ------------- Loop over all datasets
logger.info("######## Loop over all Datasets and do benchmarks ########")
for dataset in datasets_list:
    # Build benchmark for this dataset
    benchmarks = []
    for model_base in imported_models:
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
# Ensuring to generate a new file every day to keep a history of benchmark results
write_data(out_df, os.path.join(get_base_path(), get_output_result_data(), "{}_overall_benchmark_results.csv".format(
    date.today())))
