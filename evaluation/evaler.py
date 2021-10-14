# Code to eval given data according to our goals
from utils import lcer, filer
from evaluation import eval_plotter

logger = lcer.get_logger("Evaluation")


def full_data_eval(input_path, save_images=False):
    logger.info("Get Data for Input-Path {}".format(input_path))
    data = filer.read_data(input_path)

    logger.info("Get Barplots Evaluation")
    eval_plotter.start_barplot(data, save_images)
    eval_plotter.start_barplot(data, save_images, horizontal=True)

    logger.info("Get Table")
    table_data = eval_plotter.tableplot(data, save_images, "m2")

