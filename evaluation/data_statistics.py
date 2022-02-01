import pandas as pd
from data_processing.preprocessing.data_handler import load_from_files, load_datasets_information

"""
def get_statistics_from_dataframe calculates the number of users, items and rows of a dataset.
In addition to those three values, it calculates the number of features of the dataset 
(The number of features excludes the four columns: itemId, userId, rating and timestamp),
the average number of ratings per user and the min number of ratings per user. 
All calculated features are returned in a list together with the provided dataset name.
"""


def get_statistics_from_dataframe(dataframe, recsys_properties, name):
    user_list = dataframe[recsys_properties.userId_col].unique()
    users = dataframe[recsys_properties.userId_col].nunique()
    items = dataframe[recsys_properties.itemId_col].nunique()
    rows = len(dataframe.index)
    add_features = len(dataframe.drop([recsys_properties.userId_col,
                                       recsys_properties.itemId_col,
                                       recsys_properties.timestamp_col,
                                       recsys_properties.rating_col], axis=1).columns)
    avr_reviews_per_user = len(dataframe) / len(user_list)
    min_number_of_reviews, max_number_of_reviews = get_min_number_of_reviews_from_user(dataframe,
                                                                                       recsys_properties.userId_col)
    return [name, users, items, rows, add_features, avr_reviews_per_user, min_number_of_reviews, max_number_of_reviews]


# returns the number of reviews of the user with the least reviews (min number of reviews from user)
def get_min_number_of_reviews_from_user(dataframe, user_col_name):
    print("Get min number of ratings per user")
    t = dataframe[user_col_name].value_counts()

    return t.min(), t.max()


if __name__ == "__main__":
    """
    the main part of the data_statistics script calculates the statistics provided by the get_statistics_from_dataframe
    for all used dataframes. Afterwards it stores everything in a pandas Dataframe and prints it to the console. 
    The statistics are needed for the operation_autorecsys paper.
    """
    result_dataframe = pd.DataFrame(columns=['name', 'users#', 'items#', 'rows#', 'add_features#',
                                             'average_number_of_reviews_per_user#', 'min_number_of_reviews#',
                                             'max_number_of_reviews#'])
    for dataset in load_datasets_information(container_path=False):
        name, df, features, label, properties = load_from_files(dataset[0], dataset[1])
        print(name)
        statistics = pd.Series(get_statistics_from_dataframe(df, properties, name), index=result_dataframe.columns)
        result_dataframe = result_dataframe.append(statistics, ignore_index=True)

    # Export Results
    result_dataframe.to_csv("stats.csv")
