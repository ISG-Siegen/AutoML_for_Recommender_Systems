import pandas as pd
import os
from general_utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

ML_10M100k_NAME = 'movielens-10M100K'


def load_ml_10m_from_file():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-10M100K/ratings.dat'), sep='::',
                         header=0, names=['user', 'movieId', 'rating', 'timestamp'], engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column
    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_10M100k_NAME, data, features, label, recsys_properties


def load_ml_10m_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/movielens-10M100K.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_10M100k_NAME, data, features, label, recsys_properties
