import pandas as pd
import os
from utils.lcer import get_dataset_default_location
from benchmark_framework.dataset_base import RecSysProperties


def load_amazon_music_from_file():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'ratings_Digital_Music.csv'), sep=',',
                         header=0, names=['user', 'musicId', 'rating', 'timestamp'], engine='python')

    data['user'] = data.groupby(['user']).ngroup()
    data['musicId'] = data.groupby(['musicId']).ngroup()

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'musicId', 'rating', 1, 5)

    return 'amazon-music', data, features, label, recsys_properties


def load_amazon_music_from_csv():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'csv_files/amazon-music.csv'), sep=',',
                         header=True, engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_properties = RecSysProperties('userId', 'musicId', 'rating', 1, 5)

    return 'amazon-music', data, features, label, recsys_properties
