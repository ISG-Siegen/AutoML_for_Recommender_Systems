from benchmark_framework import dataset_base
from data_processing.preprocessing.preprocessing_100k import load_ml_100k_from_file, load_ml_100k_from_csv
from data_processing.preprocessing.preprocessing_1m import load_ml_1m_from_file, load_ml_1m_from_csv
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
from utils.lcer import get_logger

logger = get_logger("BenchmarkExe")

# All load functions are stored in load_file_functions and load_csv_functions list
load_file_functions = [load_ml_100k_from_file, load_ml_1m_from_file, load_ml_10m_from_file, load_ml_20m_from_file,
                       load_ml_latest_small_from_file, load_amazon_books_from_file, load_amazon_electronics_from_file,
                       load_amazon_instantvideo_from_file, load_amazon_music_from_file, load_amazon_toys_from_file,
                       load_netflix_from_file, load_yelp_from_file]

load_csv_functions = [load_ml_100k_from_csv, load_ml_1m_from_csv, load_ml_10m_from_csv, load_ml_20m_from_csv,
                      load_ml_latest_small_from_csv, load_amazon_books_from_csv, load_amazon_electronics_from_csv,
                      load_amazon_instantvideo_from_csv, load_amazon_music_from_csv, load_amazon_toys_from_csv,
                      load_netflix_from_csv, load_yelp_from_csv]


def load_file_to_csv(logger):
    logger.info("######## Store Datasets To CSV ########")

    for fn in load_file_functions:
        logger.info("Store {} To CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        data.to_csv(path_or_buf='/opt/datasets/csv_files/{}.csv'.format(name), sep=',', header=True)


def load_data_from_csv(logger):
    datasets_list = []
    logger.info("######## Load Datasets From CSV########")

    for fn in load_csv_functions:
        logger.info("Load {} From CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        datasets_list.append(dataset_base.Dataset(name, data, features, label, recsys_properties))

    return datasets_list


def load_data_from_files(logger):
    datasets_list = []
    logger.info("######## Load Datasets ########")

    for fn in load_file_functions:
        logger.info("Store {} To CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        datasets_list.append(dataset_base.Dataset(name, data, features, label, recsys_properties))

        return datasets_list


load_file_to_csv(logger)
