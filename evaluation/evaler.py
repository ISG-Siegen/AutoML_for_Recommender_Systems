# Code to eval given data according to our goals
import pandas as pd
from utils import filer
from evaluation import eval_plotter
import os
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

    for index, row in data.iterrows():
        # Check if the loop already iterated over a newer model+dataset result
        if check_dict[row["Dataset"]+row["Model"]][0] > row["timestamp"]:
            # If yes, mark the row to be dropped
            index_to_drop.append(index)
        else:
            # if no, mark last row checked for this to be dropped (if not None, i.e., initial value)
            old_index = check_dict[row["Dataset"]+row["Model"]][1]
            if old_index is not None:
                index_to_drop.append(old_index)

            # and set new timestamp, index value for checklist
            check_dict[row["Dataset"] + row["Model"]] = [row["timestamp"], index]

    # Drop and return
    return data.drop(index_to_drop).drop(columns=["timestamp"])


def eval_overall_results():
    # overall_data = merge_possible_files()
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
