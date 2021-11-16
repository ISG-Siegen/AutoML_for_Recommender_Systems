import pandas as pd
import os
from utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

AMAZON_ELECTRONICS_NAME = 'amazon-electronics'


def load_amazon_electronics_from_file():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'ratings_Electronics.csv'), sep=',',
                         header=0, names=['user', 'electronicsId', 'rating', 'timestamp'], engine='python')

    data['user'] = data.groupby(['user']).ngroup()
    data['electronicsId'] = data.groupby(['electronicsId']).ngroup()

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'electronicsId', 'rating', 1, 5)

    return AMAZON_ELECTRONICS_NAME, data, features, label, recsys_properties


def load_amazon_electronics_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/amazon-electronics.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'electronicsId', 'rating', 1, 5)

    return AMAZON_ELECTRONICS_NAME, data, features, label, recsys_properties
