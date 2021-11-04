import pandas as pd
import os
from utils.lcer import get_dataset_default_location
from benchmark_framework.dataset_base import RecSysProperties


def load_ml_20m_from_file():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'ml-20m/ratings.csv'), sep=',',
                         header=0, names=['user', 'movieId', 'rating', 'timestamp'], engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return 'movielens-20M', data, features, label, recsys_properties


def load_ml_20m_from_csv():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'csv_files/movielens-20M.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return 'movielens-20M', data, features, label, recsys_properties
