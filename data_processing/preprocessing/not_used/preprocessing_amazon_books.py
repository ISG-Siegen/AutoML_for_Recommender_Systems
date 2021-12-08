import pandas as pd
import os
from general_utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

AMAZON_BOOKS_NAME = 'amazon-books'


def load_amazon_books_from_file():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'ratings_Books.csv'), sep=',',
                         header=0, names=['user', 'bookId', 'rating', 'timestamp'], engine='python')

    data['user'] = data.groupby(['user']).ngroup()
    data['bookId'] = data.groupby(['bookId']).ngroup()

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'bookId', 'rating', 1, 5)

    return AMAZON_BOOKS_NAME, data, features, label, recsys_properties


def load_amazon_books_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/amazon-books.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'bookId', 'rating', 1, 5)

    return AMAZON_BOOKS_NAME, data, features, label, recsys_properties
