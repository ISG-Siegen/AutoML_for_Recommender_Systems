import pandas as pd
import os
from utils.lcer import get_dataset_container_path
from benchmark_framework.dataset_base import RecSysProperties

ML_1M_NAME = 'movielens-1M'


def load_ml_1m_from_file():
    ratings_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/ratings.dat'), sep='::',
                         header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/movies.dat'), sep='::',
                         header=0, names=['movieId', 'title', 'genres'], engine='python')
    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)
    movies_df = movies_df.drop(['title'], axis=1)

    user_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/movies.dat'), sep='::',
                         header=0, names=['userId', 'gender', 'age', 'occupation', 'zipCode'], engine='python')
    user_df['gender'] = user_df['gender'].replace({'F': 0, 'M': 1})

    # merge
    data = pd.merge(ratings_df, movies_df, left_on='movieId', right_on='movieId')
    data = pd.merge(data, user_df, left_on='userId', right_on='userId')

    # Drop Useless Columns
    data = data.drop(['title', 'zipCode'], axis=1)
    
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
