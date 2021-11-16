import pandas as pd
from utils.lcer import get_dataset_container_path
import os
from benchmark_framework.dataset_base import RecSysProperties

ML_100k_NAME = 'movielens-100K'


def load_ml_100k_from_file():
    """ Method to load ml100k dataset and return data, features (list of strings), and label (string) """

    # Load from Disc
    ratings_df = pd.read_csv(os.path.join(get_dataset_container_path(), 'ml-100k/u.data'), sep='\t',
                             encoding='iso-8859-1', names=['userId', 'itemId', 'rating', 'timestamp'])
    movies_df = pd.read_csv(os.path.join(get_dataset_container_path(), 'ml-100k/u.item'), sep='|',
                            encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western']
    user_df = pd.read_csv(os.path.join(get_dataset_container_path(), 'ml-100k/u.user'), sep='|',
                          encoding="iso-8859-1", header=None)
    user_df.columns = ['userId', 'age', 'gender', 'occupation', 'zip_code']

    # Merge
    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')
    rm_df = pd.merge(rm_df, user_df, left_on='userId', right_on='userId')

    # handle categorical column
    to_encode = ['occupation', 'gender']
    for col in to_encode:
        df_dummies = pd.get_dummies(rm_df[col])
        rm_df = pd.concat([rm_df, df_dummies], axis=1)

    # Drop useless columns, drop zip_code as it has multiple string-based codes which could not be encoded otherwise
    to_drop = ['title', 'releaseDate', 'imdbUrl', 'videoReleaseDate', 'itemId', 'zip_code'] + to_encode
    rm_df = rm_df.drop(to_drop, axis=1)

    # Set labels/features
    label = 'rating'
    features = list(rm_df)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_100k_NAME, rm_df, features, label, recsys_properties


def load_ml_100k_from_csv():
    data = pd.read_table(os.path.join(get_dataset_container_path(), 'csv_files/movielens-100K.csv'), sep=',',
                         header=0, engine='python')
    data = data.iloc[:, 1:]
    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 1, 5)

    return ML_100k_NAME, data, features, label, recsys_properties
