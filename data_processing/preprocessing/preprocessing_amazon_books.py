import pandas as pd
import os
from utils.lcer import get_dataset_default_location
from benchmark_framework.dataset_base import RecSysProperties


def load_amazon_books_from_file():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'ratings_Books.csv'), sep=',',
                         header=0, names=['user', 'bookId', 'rating', 'timestamp'], engine='python')

    data['user'] = data.groupby(['user']).ngroup()
    data['bookId'] = data.groupby(['bookId']).ngroup()

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'bookId', 'rating', 1, 5)

    return 'amazon-books', data, features, label, recsys_properties


def load_amazon_books_from_csv():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'csv_files/amazon-books.csv'), sep=',',
                         header=True, engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'bookId', 'rating', 1, 5)

    return 'amazon-books', data, features, label, recsys_properties
