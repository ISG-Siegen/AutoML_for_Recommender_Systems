import os
from general_utils.lcer import get_dataset_container_path
import pandas as pd
from benchmark_framework.dataset_base import RecSysProperties


def get_all_preprocess_functions():
    single_dataset_preprocessors = [preprocess_ml_100k_to_file, preprocess_ml_1m_to_file,
                                    preprocess_ml_latest_small_to_file, preprocess_yelp_to_file]

    return single_dataset_preprocessors + build_amazon_load_functions()


# ---- Specific Load Functions
# -- Movielens
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


def preprocess_ml_1m_to_file():
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

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return 'movielens-1M', data, recsys_properties


def preprocess_ml_latest_small_to_file():
    ratings_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-latest-small/ratings.csv'), sep=',',
                               header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-latest-small/movies.csv'), sep=',',
                              header=0, names=['movieId', 'title', 'genres'], engine='python')

    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)
    movies_df = movies_df.drop(['title'], axis=1)

    data = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='movieId')

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 0.5, 5)

    return 'movielens-latest-small', data, recsys_properties


# -- Amazon
def build_amazon_load_functions():
    # List of Amazon Dataset Meta-info needed to build loader
    amazon_dataset_info = [("ratings_Amazon_Instant_Video", "instantvideoId", "amazon-instantvideo"),
                           ("ratings_Toys_and_Games", "toysId", "amazon-toys"),
                           ("ratings_Digital_Music", "musicId", "amazon-music")
                           ]

    # For saving function
    load_functions_list = []

    # Build function for each combination and append to list
    for file_name, item_id_name, dataset_name in amazon_dataset_info:
        # Build load function
        def _default_amazon_preprocessor():
            data = pd.read_table(os.path.join(get_dataset_container_path(), '{}.csv'.format(file_name)), sep=',',
                                 header=0, names=['user', item_id_name, 'rating', 'timestamp'], engine='python')

            data['user'] = data.groupby(['user']).ngroup()
            data[item_id_name] = data.groupby([item_id_name]).ngroup()

            recsys_properties = RecSysProperties('userId', item_id_name, 'rating', 'timestamp', 1, 5)

            return dataset_name, data, recsys_properties

        # Add function to list
        load_functions_list.append(_default_amazon_preprocessor)

    return load_functions_list


# -- Other
def preprocess_yelp_to_file():
    data = pd.read_json(os.path.join(get_dataset_container_path(), 'yelp_training_set_review.json'), lines=True)
    data = data.rename(columns={"user_id": "user", "business_id": "itemId", "stars": "rating", "date": "timestamp"})
    data = data[['user', 'itemId', 'rating', 'timestamp']]
    data.timestamp = pd.to_datetime(data.timestamp)
    data['user'] = data.groupby(['user']).ngroup()
    data['itemId'] = data.groupby(['itemId']).ngroup()

    # TODO check if this work with the timestamp (is it a unix timestamp) - and other features
    recsys_propertys = RecSysProperties('userId', 'itemId', 'rating', 'timestamp', 1, 5)

    return 'yelp', data, recsys_propertys

# TODO add netflix with subsampling here
