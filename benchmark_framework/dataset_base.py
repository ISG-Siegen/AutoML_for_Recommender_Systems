import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, name, data: pd.DataFrame, features: list, label: str,recsys_properties=None,
                 random_state=4202021, test_size=0.25,):
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


class RecSysProperties:
    def __init__(self, userId_col: str, itemId_col: str, rating_col: str, rating_lower_bound: int,
                 rating_upper_bound: int):
        self.itemId_col = itemId_col
        self.userId_col = userId_col
        self.rating_col = rating_col
        self.rating_lower_bound = rating_lower_bound
        self.rating_upper_bound = rating_upper_bound


