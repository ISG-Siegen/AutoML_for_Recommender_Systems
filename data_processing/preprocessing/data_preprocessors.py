import os
import pandas as pd
from benchmark_framework.dataset_base import RecSysProperties
from general_utils.amazon_dataset_utils import getDF
from general_utils.yelp_dataset_utils import get_superset_of_column_names_from_file, read_and_write_file
from general_utils.netflix_dataset_utils import read_netflix_data
import math


def get_all_preprocess_functions():
    single_dataset_preprocessors = [
        preprocess_ml_100k, preprocess_ml_1m, preprocess_ml_latest_small, preprocess_yelp, preprocess_netflix,
        preprocess_food
    ]

    return single_dataset_preprocessors + build_amazon_load_functions()


# ---- Utils
def convert_date_to_timestamp(data, to_encode_columns, prefix=False):
    for col in to_encode_columns:
        df_dates = pd.to_datetime(data[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))

        if prefix:
            df_dates.name = "ts_" + col

        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dates], axis=1)
    return data


# ---- Specific Load Functions
# -- Movielens
def preprocess_ml_100k(base_path):
    """ Method to load ml100k dataset and return data, features (list of strings), and label (string) """
    # Load from Disc
    ratings_df = pd.read_csv(os.path.join(base_path, 'ml-100k/u.data'), sep='\t',
                             encoding='iso-8859-1', names=['userId', 'itemId', 'rating', 'timestamp'])
    movies_df = pd.read_csv(os.path.join(base_path, 'ml-100k/u.item'), sep='|',
                            encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western']
    user_df = pd.read_csv(os.path.join(base_path, 'ml-100k/u.user'), sep='|',
                          encoding="iso-8859-1", header=None)
    user_df.columns = ['userId', 'age', 'gender', 'occupation', 'zip_code']

    # Merge
    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')
    rm_df = pd.merge(rm_df, user_df, left_on='userId', right_on='userId')

    # handle categorical column
    to_encode_categorical = ['occupation', 'gender']
    for col in to_encode_categorical:
        df_dummies = pd.get_dummies(rm_df[col], prefix=col)
        rm_df = pd.concat([rm_df, df_dummies], axis=1)

    # Handle Dates and make them a timestamp
    rm_df = convert_date_to_timestamp(rm_df, ['releaseDate'], prefix=True)

    # Drop useless columns, drop zip_code as it has multiple string-based codes which could not be encoded otherwise
    to_drop = ['title', 'imdbUrl', 'itemId', 'zip_code', 'videoReleaseDate'] + to_encode_categorical
    rm_df = rm_df.drop(to_drop, axis=1)

    name = 'movielens-100K'
    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return name, rm_df, recsys_properties


def preprocess_ml_1m(base_path):
    ratings_df = pd.read_csv(os.path.join(base_path, 'ml-1m/ratings.dat'), sep='::',
                             header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_csv(os.path.join(base_path, 'ml-1m/movies.dat'), sep='::',
                            header=0, names=['movieId', 'title', 'genres'], engine='python', encoding='iso-8859-1')
    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)

    user_df = pd.read_csv(os.path.join(base_path, 'ml-1m/users.dat'), sep='::',
                          header=0, names=['userId', 'gender', 'age', 'occupation', 'zipCode'], engine='python')
    # gender to dummies
    df_dummies = pd.get_dummies(user_df['gender'], prefix="gender")
    user_df = pd.concat([user_df, df_dummies], axis=1)

    # merge
    data = pd.merge(ratings_df, movies_df, left_on='movieId', right_on='movieId')
    data = pd.merge(data, user_df, left_on='userId', right_on='userId')

    # Drop Useless Columns
    data = data.drop(['title', 'zipCode', 'gender'], axis=1)

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return 'movielens-1M', data, recsys_properties


def preprocess_ml_latest_small(base_path):
    ratings_df = pd.read_table(os.path.join(base_path, 'ml-latest-small/ratings.csv'), sep=',',
                               header=0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    movies_df = pd.read_table(os.path.join(base_path, 'ml-latest-small/movies.csv'), sep=',',
                              header=0, names=['movieId', 'title', 'genres'], engine='python')

    movies_df = pd.concat([movies_df.drop('genres', axis=1), movies_df.genres.str.get_dummies(sep='|')], axis=1)
    movies_df = movies_df.drop(['title'], axis=1)

    data = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='movieId')

    recsys_properties = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 0.5, 5)

    return 'movielens-latest-small', data, recsys_properties


# -- Amazon
def create_amazon_load_function(file_name, meta_file_name, dataset_name):
    def amazon_load_function_template(base_path):
        review_data = getDF(os.path.join(base_path, 'amazon', '{}.json.gz'.format(file_name)))
        meta_data = getDF(os.path.join(base_path, 'amazon', '{}.json.gz'.format(meta_file_name)))

        # hanlde review data problems
        review_data.loc[pd.isna(review_data['vote']), 'vote'] = 0

        def fix_vote_problem(vote_input):
            try:
                vote_input = int(vote_input)
                return vote_input
            except ValueError:

                return int(vote_input.replace(',', ''))

        review_data['vote'] = review_data['vote'].apply(fix_vote_problem)
        df_dummies = pd.get_dummies(review_data['verified'], prefix="verified")
        review_data = pd.concat([review_data, df_dummies], axis=1)

        # handle meta_data problems
        def fix_price_problem(price_input):

            # catch case where price is a float (some datasets)
            if isinstance(price_input, float):
                if math.isnan(price_input):
                    return -1
                else:
                    return price_input

            # catch case where price is a string (some datasets)
            price_input = price_input[1:]
            try:
                price_input = float(price_input)
                return price_input
            except ValueError:
                return -1

        meta_data['price'] = meta_data['price'].apply(fix_price_problem)

        data = review_data.drop(['image', 'reviewerName', 'style', 'reviewText', 'summary', 'reviewTime', 'verified'],
                                axis=1)

        # check if the columns are in the given data, for some reason, not all datasets have the same columns.
        # this represents teh super set of columns to drop
        to_drop = ['title', 'feature', 'description', 'imageURL', 'imageURLHighRes', 'category', 'tech1', 'tech2',
                   'also_buy', 'also_view', 'brand', 'rank', 'main_cat', 'similar_item', 'date', 'details', 'fit']
        drop_here = [d for d in to_drop if d in list(meta_data)]

        meta_data = meta_data.drop(drop_here, axis=1)

        data = pd.merge(data, meta_data, on='asin')

        data = data.rename(
            columns={'asin': 'itemId', 'reviewerID': 'userId', 'overall': 'rating', 'unixReviewTime': 'timestamp'})

        data['userId'] = data.groupby(['userId']).ngroup()
        data['itemId'] = data.groupby(['itemId']).ngroup()

        recsys_properties = RecSysProperties('userId', 'itemId', 'rating', 'timestamp', 1, 5)

        return dataset_name, data, recsys_properties

    amazon_load_function_template.__name__ = "preprocess_{}".format(dataset_name)
    return amazon_load_function_template


def build_amazon_load_functions():
    # This preprocessing script assume that the downloaded amazon datasets were moved to a folder was named "amazon"

    # List of Amazon Dataset Meta-info needed to build loader
    amazon_dataset_info = [
        ('Electronics_5', 'meta_Electronics', 'amazon-electronics'),
        ('Movies_and_TV_5', 'meta_Movies_and_TV', 'amazon-movies-and-tv'),
        ('Digital_Music_5', 'meta_Digital_Music', 'amazon-digital-music'),
        ('Toys_and_Games_5', 'meta_Toys_and_Games', 'amazon-toys-and-games'),
        ('AMAZON_FASHION_5', 'meta_AMAZON_FASHION', 'amazon-fashion'),
        ('Appliances_5', 'meta_Appliances', 'amazon-appliances'),
        ('Industrial_and_Scientific_5', 'meta_Industrial_and_Scientific', 'amazon-industrial-and-scientific'),
        ('Software_5', 'meta_Software', 'amazon-software')
    ]

    # For saving function
    load_functions_list = []

    # Build function for each combination and append to list
    for file_name, meta_file_name, dataset_name in amazon_dataset_info:
        # Build load function
        load_functions_list.append(create_amazon_load_function(file_name, meta_file_name, dataset_name))

    return load_functions_list


# -- Yelp
def preprocess_yelp(base_path):
    # We assume the downloaded yelp datasets was extracted into a folder called "yelp"
    filenames = ['yelp_academic_dataset_business', 'yelp_academic_dataset_review',
                 'yelp_academic_dataset_user']

    for filename in filenames:
        column_names = get_superset_of_column_names_from_file(os.path.join(base_path,
                                                                           ('yelp/' + filename + '.json')))
        read_and_write_file(os.path.join(base_path, ('yelp/' + filename + '.json')),
                            os.path.join(base_path, ('yelp/' + filename + '.csv')), column_names)

    business_data = pd.read_csv(os.path.join(base_path, 'yelp/yelp_academic_dataset_business.csv'))

    review_data = pd.read_csv(os.path.join(base_path, 'yelp/yelp_academic_dataset_review.csv'))

    user_data = pd.read_csv(os.path.join(base_path, 'yelp/yelp_academic_dataset_user.csv'))

    business_data = business_data[business_data.columns.drop(list(business_data.filter(regex='attributes')))]
    business_data = business_data[business_data.columns.drop(list(business_data.filter(regex='hours')))]
    business_data = business_data.drop(['name', 'address', 'city', 'postal_code', 'categories'], axis=1)

    review_data = review_data.drop(['text'], axis=1)

    user_data = user_data.drop(['name', 'friends', 'elite'], axis=1)

    review_data = review_data.rename(
        columns={'funny': 'reviev_funny', 'cool': 'review_cool', 'useful': 'review_useful'})
    business_data = business_data.rename(columns={'review_count': 'business_review_count', 'stars': 'business_stars'})
    user_data = user_data.rename(
        columns={'funny': 'user_funny', 'cool': 'user_cool', 'reviw_count': 'user_review_count'})

    data = pd.merge(review_data, business_data, on='business_id')
    data = pd.merge(data, user_data, on='user_id')

    data = data.drop(['review_id'], axis=1)
    data = data.rename(columns={"user_id": "userId", "business_id": "itemId", "stars": "rating", "date": "timestamp"})

    to_encode_dates = ['yelping_since', 'timestamp']
    for col in to_encode_dates:
        # dirty way to convert bit notation to string, hence not re-use function from above
        data[col] = data[col].apply(lambda x: x[2:-1])
        df_dates = pd.to_datetime(data[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))
        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dates], axis=1)

    categorical_culums = ['is_open', 'state']
    for col in categorical_culums:
        df_dummies = pd.get_dummies(data[col], prefix=col)
        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dummies], axis=1)

    data['userId'] = data.groupby(['userId']).ngroup()
    data['itemId'] = data.groupby(['itemId']).ngroup()

    recsys_propertys = RecSysProperties('userId', 'itemId', 'rating', 'timestamp', 1, 5)

    return 'yelp', data, recsys_propertys


# -- Netflix
def preprocess_netflix(base_path):
    # This preprocessing script assume that the downloaded archive folder was re-named to "netflix"
    filenames = ['combined_data_1',
                 'combined_data_2',
                 'combined_data_3',
                 'combined_data_4']

    read_netflix_data(filenames, base_path)

    data = pd.read_csv(os.path.join(base_path, 'netflix/fullcombined_data.csv'))
    data.columns = ['movieId', 'userId', 'rating', 'timestamp']
    # Convert timestamp
    data = convert_date_to_timestamp(data, ["timestamp"])

    movie_df = pd.read_csv(os.path.join(base_path, 'netflix/movie_titles.csv'),
                           sep=',', usecols=[0, 1], encoding='iso-8859-1', header=None)
    movie_df.columns = ['movieId', 'publish_year']

    data = pd.merge(data, movie_df, on='movieId')

    # Some moives have NULL as publish year, to handle this we simply drop these 
    data = data[data['publish_year'].notna()]

    data = data.sample(n=10000000, axis=0, random_state=42)

    recsys_propertys = RecSysProperties('userId', 'movieId', 'rating', 'timestamp', 1, 5)

    return 'netflix', data, recsys_propertys


# -- Food.com Recipe & Review Data
def preprocess_food(base_path):
    # This preprocessing script assume that the downloaded archive folder was re-named to "food_com_archive"
    interactions_df = pd.read_csv(os.path.join(base_path, "food_com_archive", "RAW_interactions.csv"))
    recipes_data_df = pd.read_csv(os.path.join(base_path, "food_com_archive", "RAW_recipes.csv"))

    # -- Preprocess Interactions DF
    interactions_df.drop(columns=["review"], inplace=True)
    interactions_df = convert_date_to_timestamp(interactions_df, ["date"])

    # -- Preprocess Recipes DF
    recipes_data_df = convert_date_to_timestamp(recipes_data_df, ["submitted"])

    # Make nutrition values their own columns
    nut_cols = recipes_data_df["nutrition"].apply(lambda x: pd.Series([float(i) for i in x[1:-1].split(',')]))
    nut_cols = nut_cols.rename(columns={x: "nutrition_" + str(x) for x in list(nut_cols)})
    recipes_data_df = pd.concat([recipes_data_df, nut_cols], axis=1)

    # Drop cols
    to_drop = ["name", "tags", "steps", "description", "ingredients", "nutrition"]
    recipes_data_df.drop(columns=to_drop, inplace=True)

    # Merge and return
    data = pd.merge(interactions_df, recipes_data_df, left_on="recipe_id", right_on="id")
    data.drop(columns=["id"], inplace=True)  # drop old id that is not needed

    recsys_properties = RecSysProperties("user_id", "recipe_id", "rating", "date", 0, 5)

    return 'foodCom', data, recsys_properties
