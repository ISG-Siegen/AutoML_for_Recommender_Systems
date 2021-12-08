# A Tool for basic file handling
import pandas as pd
import json


def read_data(input_path):
    return pd.read_csv(input_path)


def write_data(df, output_path):
    df.to_csv(output_path, index=False)


def append_data(df, output_path):
    df.to_csv(output_path, mode='a', header=False, index=False)


def write_data_json(json_data, output_path):
    with open(output_path, "w") as out_file:
        json.dump(json_data, out_file)


def read_data_json(input_path):
    with open(input_path) as in_file:
        data = json.load(in_file)
    return data
