import pandas as pd
from utils.lcer import get_dataset_default_location
import os
from pathlib import Path


def load_ml_100k():
    """ Method to load ml100k dataset and return data, features (list of strings), and label (string) """

    # Load from Disc
    ratings_df = pd.read_csv(os.path.join(get_dataset_default_location(), 'ml-100k/u.data'), sep='\t',
                             encoding='iso-8859-1', names=['userId', 'itemId', 'rating', 'timestamp'])
    movies_df = pd.read_csv(os.path.join(get_dataset_default_location(), 'ml-100k/u.item'), sep='|',
                            encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western']

    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')

    # Drop useless columns
    to_drop = ['title', 'releaseDate', 'imdbUrl', 'videoReleaseDate']  # TODO maybe also drop movieid, userid, timeid, unknown and timestamps?

    rm_df = rm_df.drop(to_drop, axis=1)

    # Set labels/features
    label = 'rating'
    features = list(rm_df)
    features.remove(label)  # this means simply all columns are features but the label column

    return rm_df, features, label


def preprocess_ml_100k():
    rm_df, features, label = load_ml_100k()

    features = pd.DataFrame(rm_df[features])
    label = pd.DataFrame(rm_df[label])

    return rm_df, features, label, rm_df.userId.unique().shape[0], rm_df.movieId.unique().shape[0]
