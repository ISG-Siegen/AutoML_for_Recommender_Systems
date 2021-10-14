import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, name, data: pd.DataFrame, features: list, label: str, random_state=4202021, test_size=0.25):
        self.name = name
        self.data = data
        self.features = features
        self.label = label

        # Split data and setup train/test data here
        df_features = self.data[self.features]
        df_label = self.data[[self.label]]
        x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=test_size,
                                                            random_state=random_state)
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
