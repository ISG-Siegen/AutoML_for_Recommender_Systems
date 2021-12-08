import os
from general_utils.lcer import get_dataset_container_path
import pandas as pd
from benchmark_framework.dataset_base import RecSysProperties


def get_all_preprocess_functions():
    single_dataset_preprocessors = [preprocess_ml_100k_to_file]

    return single_dataset_preprocessors


# ---- Specific Load Functions
def preprocess_ml_100k_to_file():
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
    to_encode_categorical = ['occupation', 'gender']
    for col in to_encode_categorical:
        df_dummies = pd.get_dummies(rm_df[col], prefix="d")
        rm_df = pd.concat([rm_df, df_dummies], axis=1)

    to_encode_dates = ['releaseDate']
    for col in to_encode_dates:
        df_dates = pd.to_datetime(rm_df[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))
        rm_df["ts_releaseDate"] = df_dates

    # Drop useless columns, drop zip_code as it has multiple string-based codes which could not be encoded otherwise
    to_drop = ['title', 'imdbUrl', 'itemId', 'zip_code', 'videoReleaseDate'] + \
              to_encode_categorical + \
              to_encode_dates
    rm_df = rm_df.drop(to_drop, axis=1)

    name = 'movielens-100K'
    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return name, rm_df, recsys_properties
