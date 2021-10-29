import pandas as pd
import os
from utils.lcer import get_dataset_default_location
from benchmark_framework.dataset_base import RecSysProperties


def load_yelp_from_file():
    data = pd.read_json(os.path.join(get_dataset_default_location(), 'yelp_training_set_review.json'), lines=True)
    data = data.rename(columns={"user_id": "user", "business_id": "itemId", "stars": "rating", "date": "timestamp"})
    data = data[['user', 'itemId', 'rating', 'timestamp']]
    data.timestamp = pd.to_datetime(data.timestamp)
    data['user'] = data.groupby(['user']).ngroup()
    data['itemId'] = data.groupby(['itemId']).ngroup()

    data = data.drop(['timestamp'], axis=1)

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_propertys = RecSysProperties('userId', 'itemId', 'rating', 1, 5)

    return 'yelp', data, features, label, recsys_propertys


def load_yelp_from_csv():
    data = pd.read_table(os.path.join(get_dataset_default_location(), 'csv_files/yelp.csv'), sep=',',
                         header=True, engine='python')

    # Set labels/features
    label = 'rating'
    features = list(data)
    features.remove(label)  # this means simply all columns are features but the label column

    recsys_propertys = RecSysProperties('userId', 'itemId', 'rating', 1, 5)

    return 'yelp', data, features, label, recsys_propertys