import pandas as pd
import os
from utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

ML_1M_NAME = 'movielens-1M'


def load_ml_1m_from_file():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/ratings.dat'), sep='::',
                         header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_1M_NAME, data, features, label, recsys_properties


def load_ml_1m_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/movielens-1M.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_1M_NAME, data, features, label, recsys_properties
