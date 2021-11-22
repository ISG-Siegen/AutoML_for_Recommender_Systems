import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.lcer import get_output_images, get_base_path
import numpy as np
import seaborn as sns
from autorank import autorank, create_report, plot_stats
from utils.catheat import heatmap

YMIN = 0
YMAX = None


def get_correct_path(file_name):
    return os.path.join(get_base_path(), get_output_images(), "{}.pdf".format(file_name))


def boxplots_per_datasets(data: pd.DataFrame, save_images, prefix=""):
    sns.catplot(x="LibraryCategory", y="RSME", col="Dataset", data=data, kind="box")

    if save_images:
        name = prefix
        name += "boxplots_per_datasets"
        plt.savefig(get_correct_path(name))
    plt.show()


def row_to_col_data_format(data):
    dataset_names = data["Dataset"].unique().tolist()
    clf_names = data["Model"].unique().tolist()
    experiment_results = pd.DataFrame(index=dataset_names, columns=clf_names)

    # Fill results df
    for index, row in data.iterrows():
        experiment_results.at[row["Dataset"], row["Model"]] = float(row["RSME"])
    experiment_results = experiment_results.apply(pd.to_numeric)

    return experiment_results


def cd_plot_and_stats_tests(data: pd.DataFrame, save_images, prefix=""):
    experiment_results = row_to_col_data_format(data)

    if len(experiment_results) < 5:
        print("SKIPPED AUTORANK EVAL DUE TO NOT ENOUGH (5) ESTIMATIONS PER MODEL")
        return

    # Do tests and get plot as well as report
    res = autorank(experiment_results, order="ascending")
    create_report(res)
    plot_stats(res)
    if save_images:
        name = prefix
        name += "autorank_plot_all_data"
        plt.savefig(get_correct_path(name))
    plt.show()

    # For more see: https://github.com/sherbold/autorank
    # Other nice-to-have features
    # latex_table(res)


def ranking_eval(data: pd.DataFrame, save_images, prefix=""):
    # Rank Input Data per Dataset
    for dataset_name in data["Dataset"].unique().tolist():
        tmp_df = data[data["Dataset"] == dataset_name]
        tmp_ranked = tmp_df["RSME"].rank()
        data.loc[data["Dataset"] == dataset_name, "RSME_RANK"] = tmp_ranked

    # -- Print Top-Average Ranked
    # Per Model
    print("######### Models Ranked - Top / bottom 10  (Average RSME, Time, Average Rank - over datasets) #########")
    rank_per_model = data.groupby(by="Model").mean()
    print(rank_per_model.sort_values(by="RSME_RANK").head(10))

    # Per Category
    print("\n ######### Categories Ranked (Average RSME, Time, Rank - over datasets and categories) #########")
    rank_per_cat = data.groupby(by="LibraryCategory").mean()
    # Get Count of model for a category (normalize by number of datasets to get true number)
    rank_per_cat["models_in_category"] = data.groupby(by="LibraryCategory").size() / data["Dataset"].nunique()
    print(rank_per_cat.sort_values(by="RSME_RANK").head(10))

    # --- Rank Plots

    # Reformat and Build data
    top_5_model_per_dataset = {}
    top_5_model_name_per_dataset = {}
    libcat_ranking_per_dataset = {}
    for dataset_name in data["Dataset"].unique().tolist():
        tmp_df = data[data["Dataset"] == dataset_name]
        tmp_df = tmp_df.sort_values(by="RSME_RANK")

        top_5_model_per_dataset[dataset_name] = tmp_df["LibraryCategory"].iloc[0:5].to_list()
        top_5_model_name_per_dataset[dataset_name] = tmp_df["Model"].iloc[0:5].to_list()

        x = tmp_df["LibraryCategory"].to_list()
        libcat_ranking_per_dataset[dataset_name] = sorted(set(x), key=x.index)

    data_for_rank_plot_models = pd.DataFrame.from_dict(top_5_model_per_dataset, orient="index",
                                                       columns=["1", "2", "3", "4", "5"])
    data_for_rank_plot_libcat = pd.DataFrame.from_dict(libcat_ranking_per_dataset, orient="index",
                                                       columns=["1", "2", "3", "4", "5"])
    # Print Top 5 model names to look for variance
    print("####### Model Names top 5 #######")
    for key, item in top_5_model_name_per_dataset.items():
        print("Dataset: {} | {}".format(key, item))

    # Top 5 Models per Dataset color coded for LibraryCategory - includes only categories in the top 5
    heatmap(data_for_rank_plot_models, leg_pos="top")
    plt.xlabel("Rank")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_images:
        name = prefix
        name += "ranking_models_per_dataset_with_categories"
        plt.savefig(get_correct_path(name))
    plt.show()

    # LibraryCategory Ranking per Dataset - includes all categories
    heatmap(data_for_rank_plot_libcat, leg_pos="top")
    plt.xlabel("Rank")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_images:
        name = prefix
        name += "ranking_categories_per_dataset"
        plt.savefig(get_correct_path(name))
    plt.show()
