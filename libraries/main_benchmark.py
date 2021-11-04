import os
from benchmark_framework import benchmarker, metrics, dataset_base
from data_processing.preprocessing.main_preprocessing import load_data_from_csv
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

from data_processing.preprocessing.preprocessing_10m import load_ml_10m_from_file, load_ml_10m_from_csv
from data_processing.preprocessing.preprocessing_20m import load_ml_20m_from_file, load_ml_20m_from_csv
from data_processing.preprocessing.preprocessing_100k_latest import load_ml_latest_small_from_file, load_ml_latest_small_from_csv
from data_processing.preprocessing.preprocessing_amazon_books import load_amazon_books_from_file, load_amazon_books_from_csv
from data_processing.preprocessing.preprocessing_amazon_electronics import load_amazon_electronics_from_file, load_amazon_electronics_from_csv
from data_processing.preprocessing.preprocessing_amazon_instantvideo import load_amazon_instantvideo_from_file, load_amazon_instantvideo_from_csv
from data_processing.preprocessing.preprocessing_amazon_music import load_amazon_music_from_file, load_amazon_music_from_csv
from data_processing.preprocessing.preprocessing_amazon_toys import load_amazon_toys_from_file, load_amazon_toys_from_csv
from data_processing.preprocessing.preprocessing_netflix import load_netflix_from_file, load_netflix_from_csv
from data_processing.preprocessing.preprocessing_yelp import load_yelp_from_file, load_yelp_from_csv
from data_processing.preprocessing.preprocessing_100k import load_ml_100k_from_file, load_ml_100k_from_csv
from data_processing.preprocessing.preprocessing_1m import load_ml_1m_from_file, load_ml_1m_from_csv

logger = get_logger("BenchmarkExe")

imported_models = [TPOTHandler, MeanPredictor, XGBoostModel,
                   FLAMLHandler, GamaHandler] + load_all_scikit_models() + load_all_surprise_models()

# ------------- Start Variables
result_data = []

# ------------- Load Datasets
datasets_list = [load_ml_10m_from_csv, load_ml_20m_from_csv,
                 load_ml_latest_small_from_csv, load_amazon_books_from_csv, load_amazon_electronics_from_csv,
                 load_amazon_instantvideo_from_csv, load_amazon_music_from_csv, load_amazon_toys_from_csv,
                 load_netflix_from_csv, load_yelp_from_csv]

# ------------- Loop over all datasets
logger.info("######## Loop over all Datasets and do benchmarks ########")
for load_function in datasets_list:
    # Build benchmark for this dataset
    benchmarks = []
    dataset = load_data_from_csv(load_function)
    for model_base in imported_models:

        benchmark = benchmarker.Benchmark(dataset, metrics.RSME(), 60, model_base())

        # Execute benchmarks for this dataset
        tmp_result_data = [(dataset.name, benchmark.model.name, benchmark.model.library_category, *benchmark.run())]

        # Evaluate Intermediate results for this dataset
        print("{}: RSME of {} | Time take {}".format(tmp_result_data[0][1], tmp_result_data[0][3], tmp_result_data[0][4]))

        # Add result of this dataset to full collection

        # ------------- Output Data as results file
        logger.info("######## Export Result data ########")
        out_df = pd.DataFrame(tmp_result_data)
        # Ensuring to generate a new file every day to keep a history of benchmark results
        write_data(out_df, os.path.join(get_base_path(), get_output_result_data(), "{}_overall_benchmark_results.csv".format(date.today())))
        print('##########Data Written############')
