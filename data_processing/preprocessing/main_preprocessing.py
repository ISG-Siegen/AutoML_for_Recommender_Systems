from benchmark_framework import dataset_base
from data_processing.preprocessing.preprocessing_100k import load_ml_100k_from_file, load_ml_100k_from_csv, ML_100k_NAME
from data_processing.preprocessing.preprocessing_1m import load_ml_1m_from_file, load_ml_1m_from_csv, ML_1M_NAME
# from data_processing.preprocessing.preprocessing_10m import load_ml_10m_from_file, load_ml_10m_from_csv, \
# ML_10M100k_NAME
# from data_processing.preprocessing.preprocessing_20m import load_ml_20m_from_file, load_ml_20m_from_csv, ML_20M_NAME
from data_processing.preprocessing.preprocessing_100k_latest import load_ml_latest_small_from_file, \
    load_ml_latest_small_from_csv, ML_LATESTSMALL_NAME
# from data_processing.preprocessing.preprocessing_amazon_books import load_amazon_books_from_file, \
#    load_amazon_books_from_csv, AMAZON_BOOKS_NAME
# from data_processing.preprocessing.preprocessing_amazon_electronics import load_amazon_electronics_from_file, \
#    load_amazon_electronics_from_csv, AMAZON_ELECTRONICS_NAME
from data_processing.preprocessing.preprocessing_amazon_instantvideo import load_amazon_instantvideo_from_file, \
    load_amazon_instantvideo_from_csv, AMAZON_INSTANTVIDEO_NAME
from data_processing.preprocessing.preprocessing_amazon_music import load_amazon_music_from_file, \
    load_amazon_music_from_csv, AMAZON_MUSIC_NAME
from data_processing.preprocessing.preprocessing_amazon_toys import load_amazon_toys_from_file, \
    load_amazon_toys_from_csv, AMAZON_TOYS_NAME
# from data_processing.preprocessing.preprocessing_netflix import load_netflix_from_file, load_netflix_from_csv, \
#    NETFLIX_NAME
from data_processing.preprocessing.preprocessing_yelp import load_yelp_from_file, load_yelp_from_csv, YELP_NAME
from general_utils.lcer import get_logger

logger = get_logger("Preprocessing")

# All load functions are stored in load_file_functions and load_csv_functions list
LOAD_FILE_FUNCTIONS = [load_ml_100k_from_file, load_ml_1m_from_file,
                       load_ml_latest_small_from_file, load_yelp_from_file,
                       load_amazon_instantvideo_from_file, load_amazon_music_from_file, load_amazon_toys_from_file,

                       # Remove for now
                       # load_ml_10m_from_file, load_ml_20m_from_file,
                       #   load_netflix_from_file, load_amazon_books_from_file, load_amazon_electronics_from_file,
                       ]

# ------------ Relevant Dicts for Data Processing (must have same order)
LOAD_CSV_FUNCTIONS = [
    load_ml_100k_from_csv, load_ml_1m_from_csv,
    load_ml_latest_small_from_csv, load_amazon_instantvideo_from_csv, load_amazon_music_from_csv,
    load_amazon_toys_from_csv, load_yelp_from_csv

    # Removed because too big for now: load_ml_20m_from_csv, load_netflix_from_csv, load_ml_10m_from_csv
    # load_amazon_books_from_csv, load_amazon_electronics_from_csv
]
DATASET_NAMES = [
    ML_100k_NAME, ML_1M_NAME, ML_LATESTSMALL_NAME, AMAZON_INSTANTVIDEO_NAME, AMAZON_MUSIC_NAME,
    AMAZON_TOYS_NAME, YELP_NAME

    # Not used as above: ML_20M_NAME, NETFLIX_NAME, AMAZON_BOOKS_NAME, AMAZON_ELECTRONICS_NAME, ML_10M100k_NAME
]


# ------------ Code


def load_file_to_csv():
    logger.info("######## Store Datasets To CSV ########")

    for fn in LOAD_FILE_FUNCTIONS:
        logger.info("Store {} To CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        data.to_csv(path_or_buf='/opt/datasets/csv_files/{}.csv'.format(name), sep=',', header=True)


def load_all_data_from_csv():
    datasets_list = []
    logger.info("######## Load Datasets From CSV########")

    for fn in LOAD_CSV_FUNCTIONS:
        logger.info("Load {} From CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        datasets_list.append(dataset_base.Dataset(name, data, features, label, recsys_properties))

    return datasets_list


def load_data_from_files():
    datasets_list = []
    logger.info("######## Load Datasets ########")

    for fn in LOAD_FILE_FUNCTIONS:
        logger.info("Store {} To CSV".format(fn))
        name, data, features, label, recsys_properties = fn()
        datasets_list.append(dataset_base.Dataset(name, data, features, label, recsys_properties))

        return datasets_list


def get_dataset_load_functions():
    return LOAD_CSV_FUNCTIONS
