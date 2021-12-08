import pandas as pd
import os
from general_utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

AMAZON_TOYS_NAME = 'amazon-toys'


def load_amazon_toys_from_file():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'ratings_Toys_and_Games.csv'), sep=',',
                         header=0, names=['user', 'toysId', 'rating', 'timestamp'], engine='python')

    data['user'] = data.groupby(['user']).ngroup()
    data['toysId'] = data.groupby(['toysId']).ngroup()

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'toysId', 'rating', 1, 5)

    return AMAZON_TOYS_NAME, data, features, label, recsys_properties


def load_amazon_toys_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/amazon-toys.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'toysId', 'rating', 1, 5)

    return AMAZON_TOYS_NAME, data, features, label, recsys_properties
