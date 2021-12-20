import os
import sys
import json
import glob
import pandas as pd

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from general_utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties
from general_utils.lcer import get_logger
from data_processing.preprocessing.data_preprocessors import get_all_preprocess_functions

logger = get_logger("data_preprocessor")


# --- Utils
def save_to_files(name, data_df, recsys_properties):
    logger.info("######## Store Datasets {} to CSV and its meta-data to JSON ########".format(name))

    # Rename columns to standardized format
    data_df = recsys_properties.transform_dataset(data_df)
    file_path_csv = os.path.join(get_dataset_container_path(), "preprocessed_data/{}.csv".format(name))
    file_path_json = os.path.join(get_dataset_container_path(), "preprocessed_data/{}.json".format(name))

    # Add meta data as empty columns
    data_df.to_csv(path_or_buf=file_path_csv, sep=',', header=True, index=False)

    # Add meta_data as own file
    meta_data = {
        "dataset_name": name,
        "rating_lower_bound": recsys_properties.rating_lower_bound,
        "rating_upper_bound": recsys_properties.rating_upper_bound
    }
    with open(file_path_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


def load_from_files(path_to_file_csv, path_to_file_json):
    # Read Data
    df = pd.read_csv(path_to_file_csv)
    with open(path_to_file_json) as json_file:
        meta_data = json.load(json_file)
    name = meta_data["dataset_name"]

    # Build RecSys Properties
    properties = RecSysProperties(RecSysProperties._default_userId_col_name,
                                  RecSysProperties._default_itemId_col_col_name,
                                  RecSysProperties._default_rating_col_name,
                                  RecSysProperties._default_timestamp_col_name,
                                  meta_data["rating_lower_bound"], meta_data["rating_upper_bound"])

    # Define features and label
    label = properties.rating_col
    features = list(df)
    features.remove(label)

    return name, df, features, label, properties


# --- Function to be used to preprocess and load preprocessed data
def get_all_dataset_names():
    return [x[2] for x in load_datasets_information()]


def load_datasets_information():
    dir_path_csv = os.path.join(get_dataset_container_path(), "preprocessed_data/*.csv")
    dir_path_json = os.path.join(get_dataset_container_path(), "preprocessed_data/*.json")

    data_paths = glob.glob(dir_path_csv)
    meta_data_paths = glob.glob(dir_path_json)

    # Merge files
    path_tuples = []
    for csv_path in data_paths:
        csv_name = csv_path[:-4]
        for json_path in meta_data_paths:
            json_name = json_path[:-5]
            if csv_name == json_name:
                path_tuples.append((csv_path, json_path, os.path.basename(csv_name)))
                break

    # Validate number of pairs
    l_dp = len(data_paths)
    l_mdp = len(meta_data_paths)
    l_pt = len(path_tuples)
    if l_pt < l_mdp or l_pt < l_dp:
        logger.warning("Found more files in the preprocessed data directory than file pairs: " +
                       "Pairs {}, CSV files: {}, JSON files: {}".format(l_pt, l_dp, l_mdp))

    # Validate correctness of merge
    for paths_tuple in path_tuples:
        try:
            assert paths_tuple[0][:-4] == paths_tuple[1][:-5]
        except AssertionError:
            raise ValueError("Some data files have wrongly configured names: {} vs. {}".format(paths_tuple[0],
                                                                                               paths_tuple[1]))

    return path_tuples


def preprocess_all_datasets():
    preprocessors = get_all_preprocess_functions()

    n_preprocessors = len(preprocessors)
    for idx, fn in enumerate(preprocessors, 1):
        logger.info("Start Preprocessing: {} [{}/{}]".format(fn.__name__, idx, n_preprocessors))
        # Preprocess and save results to csv
        save_to_files(*fn())


if __name__ == "__main__":
    preprocess_all_datasets()
