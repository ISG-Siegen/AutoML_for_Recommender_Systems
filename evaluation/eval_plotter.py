import matplotlib.pyplot as plt
import pandas as pd
import os
from general_utils.lcer import get_output_images, get_base_path, get_output_result_tables
import seaborn as sns
from autorank import autorank, create_report, plot_stats
from general_utils.catheat import heatmap

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
    # Rank Input Data per Dataset (uses average rank, lower is better)
    #   The input values (for the rmse) can have nan values representing a failure as a result of running into a limit.
    #   To counteract this, anything that did run into a limit gets ranked last (if multiple all get the same rank)
    #   By default, equal values are assigned a rank that is the average of the ranks of those values.
    for dataset_name in data["Dataset"].unique().tolist():
        tmp_df = data[data["Dataset"] == dataset_name]
        tmp_ranked = tmp_df["RSME"].rank(na_option="bottom")
        data.loc[data["Dataset"] == dataset_name, "RMSE_RANK"] = tmp_ranked

    # -- Print Top-Average Ranked
    # Per Model
    print("######### Models Ranked - Top 10  (Average values for: Rank, RMSE - over datasets) #########")
    rank_per_model = data.groupby(by="Model").mean()
    res_table_top_10_models = rank_per_model[["RMSE_RANK", "RSME"]].sort_values(by="RMSE_RANK").head(10)
    print(res_table_top_10_models)

    # Per Category
    print("\n ######### Categories Ranked (Average values for: Rank, failed, models_in_category" +
          "- over datasets and categories) #########")
    rank_per_cat = data.groupby(by="LibraryCategory").mean()
    # Get Count of model for a category (normalize by number of datasets to get true number)
    rank_per_cat["algorithm_in_category"] = data.groupby(by="LibraryCategory").size() / data["Dataset"].nunique()
    res_table_rank_per_cat = rank_per_cat[["RMSE_RANK", "failed", "algorithm_in_category"]].sort_values(by="RMSE_RANK")
    print(res_table_rank_per_cat)

    # -- Save Tables
    res_table_top_10_models.to_csv(os.path.join(get_base_path(), get_output_result_tables(), "rank_top_10_models.csv"))
    res_table_rank_per_cat.to_csv(os.path.join(get_base_path(), get_output_result_tables(), "rank_per_cat.csv"))

    # --- Rank Plots
    # Reformat and Build data
    top_5_category_per_dataset = {}
    top_5_stats_per_dataset = {}
    libcat_ranking_per_dataset = {}
    top_5_stats_per_cat_per_dataset = {}
    for dataset_name in data["Dataset"].unique().tolist():
        tmp_df = data[data["Dataset"] == dataset_name]
        tmp_df = tmp_df.sort_values(by="RMSE_RANK")

        # Get Values for Heatmap colors
        top_5_category_per_dataset[dataset_name] = tmp_df["LibraryCategory"].iloc[0:5].to_list()
        # Top 5 unique
        x = tmp_df["LibraryCategory"].to_list()
        libcat_ranking_per_dataset[dataset_name] = sorted(set(x), key=x.index)

        # Get Stats for annotation for Top 5 algorithms and top 5 categories
        top_5_stats_per_dataset[dataset_name] = tmp_df[["Model", "RSME"]].iloc[0:5].values.tolist()

        # Get index for sorted set of categories to find correct values
        top_5_cats = set()
        top_5_cat_index = []
        for i, cat in enumerate(x, 0):
            if cat not in top_5_cats:
                top_5_cats.add(cat)
                top_5_cat_index.append(i)
        top_5_stats_per_cat_per_dataset[dataset_name] = [tmp_df[["Model", "RSME"]].iloc[i].values.tolist()
                                                         for i in top_5_cat_index]

    # Build heatmap colors for category rankings
    nr_cats = len(data["LibraryCategory"].unique().tolist())
    data_for_rank_plot_libcat = pd.DataFrame.from_dict(libcat_ranking_per_dataset, orient="index",
                                                       columns=[str(i) for i in range(1, nr_cats + 1)])
    # Build heatmap color data for top algorithms per dataset
    data_for_rank_plot_models = pd.DataFrame.from_dict(top_5_category_per_dataset, orient="index",
                                                       columns=["1", "2", "3", "4", "5"])

    # Add annotation
    def replace_names_for_paper(specific_name):
        # A hand made function to change the internal names of algorithms to the names used in the paper

        # filter special cases
        if specific_name == "H2O_AutoML":
            return "H20AutoML"
        if "_" in specific_name:
            n_prefix, n_postfix = specific_name.split("_")

            # Change postfix
            if n_postfix == "SingularValueDecompositionAlgorithm":
                n_postfix = "SVD"

            # Filter prefix
            if n_prefix in ["lenskit", "Surprise", "SciKit"]:
                if n_prefix == "lenskit":
                    return n_postfix + " (LensKit)"
                elif n_prefix == "SciKit":
                    return n_postfix + " (SciKit)"
                else:
                    return n_postfix + " (Surprise)"
            # Filter postfix
            if n_postfix in ["Regressor", "TabularRegressor"]:
                return n_prefix

            if n_prefix == "ConstantPredictor":
                return n_postfix + " (ConstantPredictor)"

        return specific_name

    # Top 5 Models per Dataset color coded for LibraryCategory - includes only categories in the top 5
    top_5_algo_cats_annot = [
        ["{:.4f} \n {}".format(x[1], replace_names_for_paper(x[0])) for x in top_5_stats_per_dataset[d_name]] for
        d_name in data_for_rank_plot_models.index.tolist()]
    heatmap(data_for_rank_plot_models, leg_pos="top", annot=top_5_algo_cats_annot, fmt="s", annot_kws={"size": "small"})
    plt.xlabel("Rank")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_images:
        name = prefix
        name += "ranking_models_per_dataset_with_categories"
        # plt.savefig(get_correct_path(name))
    plt.show()

    # LibraryCategory Ranking per Dataset - includes all categories
    top_5_cat_annot = [
        ["{:.4f} \n {}".format(x[1], replace_names_for_paper(x[0])) for x in top_5_stats_per_cat_per_dataset[d_name]]
        for d_name in data_for_rank_plot_models.index.tolist()]
    heatmap(data_for_rank_plot_libcat, leg_pos="top", annot=top_5_cat_annot, fmt="s", annot_kws={"size": "small"})
    plt.xlabel("Rank")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_images:
        name = prefix
        name += "ranking_categories_per_dataset"
        # plt.savefig(get_correct_path(name))
    plt.show()
