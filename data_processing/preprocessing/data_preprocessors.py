import os
from general_utils.lcer import get_dataset_container_path
import pandas as pd
from benchmark_framework.dataset_base import RecSysProperties
from general_utils.amazon_dataset_utils import getDF


def get_all_preprocess_functions():
    single_dataset_preprocessors = [preprocess_ml_100k, preprocess_ml_1m,
                                    preprocess_ml_latest_small, preprocess_yelp]

    return single_dataset_preprocessors + build_amazon_load_functions()


# ---- Specific Load Functions
# -- Movielens
def preprocess_ml_100k():
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


def preprocess_ml_1m():
    ratings_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/ratings.dat'), sep='::',
                               header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/movies.dat'), sep='::',
                              header=0, names=['movieId', 'title', 'genres'], engine='python')
    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)
    movies_df = movies_df.drop(['title'], axis=1)

    user_df = pd.read_table(os.path.join(get_dataset_container_path(), 'ml-1m/movies.dat'), sep='::',
                            header=0, names=['userId', 'gender', 'age', 'occupation', 'zipCode'], engine='python')
   # TODO gender to dummies
    user_df['gender'] = user_df['gender'].replace({'F': 0, 'M': 1})

    # merge
    data = pd.merge(ratings_df, movies_df, left_on='movieId', right_on='movieId')
    data = pd.merge(data, user_df, left_on='userId', right_on='userId')

    # Drop Useless Columns
    data = data.drop(['title', 'zipCode'], axis=1)

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return 'movielens-1M', data, recsys_properties


def preprocess_ml_latest_small():
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
    amazon_dataset_info = [('Electronics_5', 'meta_Electronics', 'amazon-electronics'),
                           ("Movies_and_TV_5", 'meta_Movies_and_TV', 'amazon-movies-and-tv'),
                           ('Digital_Music_5', 'meta_Digital_Music', 'amazon-digital-music'),
                           ('Toys_and_Games_5', 'meta_Toys_and_Games', 'amazon-toys-and-games')]

    # For saving function
    load_functions_list = []

    # Build function for each combination and append to list
    for file_name, meta_file_name, dataset_name in amazon_dataset_info:
        # Build load function
        def _default_amazon_preprocessor():
            review_data = getDF(os.path.join(get_dataset_container_path(), '{}.csv'.format(file_name)))
            meta_data =getDF(os.path.join(get_dataset_container_path(), '{}.csv'.format(meta_file_name)))

            data = review_data.drop(['image', 'reviewerName', 'style', 'reviewerText', 'summary', 'reviewTime'], axis=1)
            meta_data = meta_data.drop(['title', 'feature', 'description', 'imageURL', 'imageURLHighRes'
                                        'also_viewed', 'tech1', 'tech2', 'similar', 'categories'], axis=1)

            data = pd.merge(data, meta_data, on='asin')

            data.rename(
                columns={'asin': 'itemId', 'reviewerId': 'userId', 'overall': 'rating', 'unixReviewTime': 'timestamp'})

            data['user'] = data.groupby(['user']).ngroup()
            data['itemId'] = data.groupby(['itemId']).ngroup()

            recsys_properties = RecSysProperties('userId', 'itemId', 'rating', 'timestamp', 1, 5)

            return dataset_name, data, recsys_properties

        _default_amazon_preprocessor.__name__ = "preprocess_{}".format(dataset_name)

        # Add function to list
        load_functions_list.append(_default_amazon_preprocessor)

    return load_functions_list


# -- Yelp
def preprocess_yelp():
    business_data = pd.read_json(os.path.join(get_dataset_container_path(), 'business.json'), lines=True)
    review_data = pd.read_json(os.path.join(get_dataset_container_path(), 'review.json'), lines=True)
    user_data = pd.read_json(os.path.join(get_dataset_container_path(), 'user.json'), lines=True)
    tip_data = pd.read_json(os.path.join(get_dataset_container_path(), 'tip.json'), lines=True)

    #TODO check postal code, string city
    business_data = business_data.drop(['name', 'address', 'city', 'postal code', 'categories', 'hours',
                                        'attributes'], axis=1)

    review_data = review_data.drop(['text'], axis=1)

    user_data = user_data.drop(['name', 'friends', 'elite'], axis=1)

    # TODO check data after merge
    tip_data = tip_data.drop(['text'], axis=1)
    tip_data = tip_data.rename(columns={'date': 'tip_date'})

    data = pd.merge(review_data, business_data, on='business_id')
    data = pd.merge(data, user_data, on='user_id')
    data = pd.merge(data, tip_data, on='user_id')

    data = data.drop(['review_id'], axis=1)
    data = data.rename(columns={"user_id": "userId", "business_id": "itemId", "stars": "rating", "date": "timestamp"})

    to_encode_dates = ['date', 'tip_date', 'yelping_since']
    for col in to_encode_dates:
        df_dates = pd.to_datetime(data[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))
        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dates], axis=1)

    categorical_culums = ['is_open', 'state']
    for col in categorical_culums:
        df_dummies = pd.get_dummies(data[col], prefix="d")
        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dummies], axis=1)

    data['user'] = data.groupby(['user']).ngroup()
    data['itemId'] = data.groupby(['itemId']).ngroup()

    recsys_propertys = RecSysProperties('userId', 'itemId', 'rating', 'timestamp', 1, 5)

    return 'yelp', data, recsys_propertys

# TODO add netflix with subsampling here
