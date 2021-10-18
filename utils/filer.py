# A Tool for basic file handling
import pandas as pd


def read_data(input_path):
    return pd.read_csv(input_path)


def write_data(df, output_path):
    df.to_csv(output_path, index=False)
