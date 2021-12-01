import pandas as pd
import os
from utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

ML_LATESTSMALL_NAME = 'movielens-latest-small'


def load_ml_latest_small_from_file():
    ratings_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-latest-small/ratings.csv'), sep=',',
                        header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-latest-small/movies.csv'), sep=',',
                         header=0, names=['movieId', 'title', 'genres'], engine='python')

    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)
    movies_df = movies_df.drop(['title'], axis=1)

    data = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='movieId')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_LATESTSMALL_NAME, data, features, label, recsys_properties


def load_ml_latest_small_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/movielens-latest-small.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_LATESTSMALL_NAME, data, features, label, recsys_properties
