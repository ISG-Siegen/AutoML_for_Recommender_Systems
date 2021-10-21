# Code to eval given data according to our goals
from utils import filer
from evaluation import eval_plotter
import os
from utils.lcer import get_logger, get_output_result_data, get_base_path
from datetime import date

logger = get_logger("Evaluation")


def example_data_eval(input_path, save_images=True):
    logger.info("Get Data for Input-Path {}".format(input_path))
    data = filer.read_data(input_path)

    get_basic_plots(data, save_images, "m2")


def get_basic_plots(data, save_images, baseline_model_name, prefix=""):
    logger.info("Get Barplots Evaluation")
    # eval_plotter.start_barplot(data, save_images, prefix=prefix)
    eval_plotter.start_barplot(data, save_images, horizontal=True, prefix=prefix)

    logger.info("Get Table")
    table_data = eval_plotter.tableplot(data, save_images, baseline_model_name,prefix=prefix)

    return table_data


def eval_overall_results():
    overall_data = filer.read_data(os.path.join(get_base_path(), get_output_result_data(), "{}_overall_benchmark_results.csv".format(date.today())))

    # Filter too large errors for model that did not converge with default values
    overall_data = overall_data[overall_data["RSME"] <= 2]

    # FOr each dataset
    for dataset in overall_data["Dataset"].unique().tolist():
        logger.info("Get Barplots Evaluation for dataset {}".format(dataset))
        image_name_prefix = "{}_".format(dataset)

        result_subset = overall_data[overall_data["Dataset"] == dataset]
        get_basic_plots(result_subset[["Model", "RSME"]], True, "SciKit_RandomForestRegressor", prefix=image_name_prefix)

        eval_plotter.aggregated_barplot(result_subset[["Model", "LibraryCategory", "RSME"]], True, "LibraryCategory",
                                        prefix=image_name_prefix)


    # TODO potentially for all datasets at once


if __name__ == "__main__":
    eval_overall_results()
