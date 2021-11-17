# Code to eval given data according to our goals
import os
import sys
import itertools

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Default imports
import pandas as pd
from utils import filer
from evaluation import eval_plotter
from utils.lcer import get_logger, get_output_result_data, get_base_path
import numpy as np
from collections import defaultdict

logger = get_logger("Evaluation")
pd.set_option("display.max_rows", None, "display.max_columns", None)


def filter_too_large_errors(df, dataset_names):
    for dataset in dataset_names:
        tmp_df = df[df["Dataset"] == dataset]["RSME"]
        # Code adapted from https://datascience.stackexchange.com/a/57199
        Q1 = tmp_df.quantile(0.25)
        Q3 = tmp_df.quantile(0.75)
        IQR = Q3 - Q1

        # Set all values lower than the upper whisker to nan and then drop them
        df.loc[(df["Dataset"] == dataset) & (df["RSME"] > Q3 + 1.5 * IQR), "RSME"] = np.nan

    return df.dropna()


def select_newest_subset(data):
    # Filter all older runs of models on datasets
    check_dict = defaultdict(lambda: [0, None])
    index_to_drop = []

    # Get list of model and dataset names that are valid (depending on our current run overhead data)
    run_overhead_data = filer.read_data_json(os.path.join(get_base_path(), get_output_result_data(),
                                                          "run_overhead_data.json"))
    valid_datasets = set(run_overhead_data["used_dataset_names"])
    valid_models = set([model_name for lib in run_overhead_data["used_libraries_names"]
                        for model_name in run_overhead_data[lib]["model_names"]])

    for index, row in data.iterrows():
        # Check if the loop already iterated over a newer model+dataset result
        dataset_name = row["Dataset"]
        model_name = row["Model"]

        if (dataset_name not in valid_datasets) or (model_name not in valid_models) \
                or (check_dict[dataset_name + model_name][0] > row["timestamp"]):
            # If yes, mark the row to be dropped
            index_to_drop.append(index)
        else:
            # if no, mark last row checked for this to be dropped (if not None, i.e., initial value)
            old_index = check_dict[dataset_name + model_name][1]
            if old_index is not None:
                index_to_drop.append(old_index)

            # and set new timestamp, index value for checklist
            check_dict[row["Dataset"] + row["Model"]] = [row["timestamp"], index]

    # Drop and return
    dropped_data = data.drop(index_to_drop).drop(columns=["timestamp"])

    # Check for how many values are still missing
    valid_pairs = set(itertools.product(valid_models, valid_datasets))
    print("Total amount of evaluation pairs: {}".format(len(valid_pairs)))

    for index, row in dropped_data[["Dataset", "Model"]].iterrows():
        tmp_pair = (row["Model"], row["Dataset"])
        valid_pairs.remove(tmp_pair)

    print("Amount of evaluation pairs left to do: {}".format(len(valid_pairs)))

    return dropped_data


def eval_overall_results():
    overall_data = filer.read_data(os.path.join(get_base_path(), get_output_result_data(),
                                                "overall_benchmark_results.csv"))
    overall_data = select_newest_subset(overall_data)

    # ----- Filter too large errors for model that did not converge with default values
    dataset_names = overall_data["Dataset"].unique().tolist()

    overall_data_filtered = filter_too_large_errors(overall_data, dataset_names)

    # Some Plots over all Datasets
    eval_plotter.boxplots_per_datasets(overall_data_filtered[["Dataset", "LibraryCategory", "RSME"]], True)
    eval_plotter.cd_plot_and_stats_tests(overall_data_filtered, True)
    eval_plotter.ranking_eval(overall_data_filtered, True)


if __name__ == "__main__":
    eval_overall_results()
