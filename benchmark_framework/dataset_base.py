import pandas as pd
from sklearn.model_selection import train_test_split


class RecSysProperties:
    _default_userId_col_name = "userId"
    _default_itemId_col_col_name = "itemId"
    _default_timestamp_col_name = "timestamp"
    _default_rating_col_name = "rating"

    def __init__(self, userId_col: str, itemId_col: str, rating_col: str, timestamp_col: str, rating_lower_bound: float,
                 rating_upper_bound: float):
        self.itemId_col = itemId_col
        self.userId_col = userId_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.rating_lower_bound = rating_lower_bound
        self.rating_upper_bound = rating_upper_bound

        self.user_num = self.item_num = 0

        # Default values for column names

    def set_num_values(self, df):
        """Function to set the number of users and items for usage of certain libraries"""
        self.user_num = df[self.userId_col].nunique()
        self.item_num = df[self.itemId_col].nunique()

    def get_num_values(self):
        return self.user_num, self.item_num

    def transform_dataset(self, data_df):
        """Function to transform existing dataset to correct format for saving it"""

        return data_df.rename(columns={
            self.itemId_col: self._default_itemId_col_col_name,
            self.userId_col: self._default_userId_col_name,
            self.timestamp_col: self._default_timestamp_col_name,
            self.rating_col: self._default_rating_col_name
        })


class Dataset:
    def __init__(self, name, data: pd.DataFrame, features: list, label: str, recsys_properties: RecSysProperties,
                 random_state=4202021, test_size=0.25):
        self.name = name
        self.data = data
        self.features = features
        self.label = label
        self.recsys_properties = recsys_properties

        # Split data and setup train/test data here
        df_features = self.data[self.features]
        df_label = self.data[[self.label]]
        x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=test_size,
                                                            random_state=random_state)
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

        # Set user/item number
        self.recsys_properties.set_num_values(self.data)


class DummyDataset:
    def __init__(self):
        self.features = ["dummy_feature"]
