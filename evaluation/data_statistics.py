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
    avr_reviews_per_user = len(dataframe)/len(user_list)
    min_number_of_reviews_from_user = get_min_number_of_reviews_from_user(dataframe,
                                                                          user_list,
                                                                          recsys_properties.userId_col)
    return [name, users, items, rows, add_features, avr_reviews_per_user, min_number_of_reviews_from_user]


# returns the number of reviews of the user with the least reviews (min number of reviews from user)
def get_min_number_of_reviews_from_user(dataframe, users, user_col_name):
    min_ratings = float('inf')
    for user in users:
        number_of_ratings = len(dataframe[dataframe[user_col_name] == user])
        if number_of_ratings < min_ratings:
            min_ratings = number_of_ratings
    return min_ratings


if __name__ == "__main__":
    """
    the main part of the data_statistics script calculates the statistics provided by the get_statistics_from_dataframe
    for all used dataframes. Afterwards it stores everything in a pandas Dataframe and prints it to the console. 
    The statistics are needed for the operation_autorecsys paper.
    """
    result_dataframe = pd.DataFrame(columns=['name', 'users#', 'items#', 'rows#', 'add_features#',
                                             'average_number_of_reviews_per_user#', 'user_with_min_number_of_reviews#'])
    for dataset in load_datasets_information():
        name, df, features, label, properties = load_from_files(dataset[0], dataset[1])
        statistics = pd.Series(get_statistics_from_dataframe(df, properties, name), index=result_dataframe.columns)
        result_dataframe = result_dataframe.append(statistics, ignore_index=True)
    print(result_dataframe)
