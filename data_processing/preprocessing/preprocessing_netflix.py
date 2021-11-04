import pandas as pd
import os
from utils.lcer import get_dataset_default_location
from benchmark_framework.dataset_base import RecSysProperties


def load_netflix_from_file():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'NetflixRatings.csv'), sep=',',
                         header=0, names=['itemId', 'user', 'rating', 'timestamp'], engine='python')
    data = data[['user', 'itemId', 'rating', 'timestamp']]

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'itemId', 'rating', 1, 5)

    return 'netflix', data, features, label, recsys_properties


def load_netflix_from_csv():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'csv_files/netflix.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'itemId', 'rating', 1, 5)

    return 'netflix', data, features, label, recsys_properties
