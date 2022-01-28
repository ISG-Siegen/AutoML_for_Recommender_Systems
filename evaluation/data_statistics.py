import pandas as pd
from data_processing.preprocessing.data_handler import load_from_files, load_datasets_information
"""
def get_statistics_from_datafram calculates the number of users, items and rows of a dataset.
In addition to those three values, it calculates the number of features of the dataset. 
(The number of features excludes the four columns: itemId, userId, rating and timestamp)
All calculated features are returned in a list together with the provided dataset name.
"""
def get_statistics_from_dataframe(dataframe, recsys_properties, name):
    users = dataframe[recsys_properties.userId_col].nunique()
    items = dataframe[recsys_properties.itemId_col].nunique()
    rows = len(dataframe.index)
    add_features = len(dataframe.drop([recsys_properties.userId_col,
                                       recsys_properties.itemId_col,
                                       recsys_properties.timestamp_col,
                                       recsys_properties.rating_col], axis=1).columns)
    return [name, users, items, rows, add_features]


if __name__ == "__main__":
    """
    the main part of the data_statistics script calculates the statistics provided by the get_statistics_from_dataframe
    for all used dataframes. Afterwards it stores everything in a pandas Dataframe and prints it to the console. 
    The statistics are needed for the operation_autorecsys paper.
    """
    result_dataframe = pd.DataFrame(columns=['name', 'users#', 'items#', 'rows#', 'add_features#'])
    for dataset in load_datasets_information():
        name, df, features, label, properties = load_from_files(dataset[0], dataset[1])
        statistics = pd.Series(get_statistics_from_dataframe(df, properties, name), index=result_dataframe.columns)
        result_dataframe = result_dataframe.append(statistics, ignore_index=True)
    print(result_dataframe)
