# A Tool for handling logging and configurations
import logging
from configparser import ConfigParser
import os
import pathlib
from benchmark_framework.metrics import NAME_TO_METRIC

local_config = ConfigParser()
local_config.read(os.path.join(os.path.dirname(__file__), "../config.ini"))

# Disable useless warnings from Matplotlib
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


# --------------- Logger Code
def get_logger(log_name):
    # Logger setup
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    )

    return logging.getLogger(log_name)


# --------------- Path management
def get_base_path():
    return pathlib.Path(__file__).parent.parent.resolve()


# --------------- Config Code
def get_config(category, entry):
    return local_config[category][entry]


def get_dataset_local_path():
    return get_config("datasets", "local_path")


def get_dataset_container_path():
    return get_config("datasets", "container_path")


def get_default_metric():
    return NAME_TO_METRIC[get_config("defaults", "metric")]


def get_output_result_data():
    return get_config("output", "result_data")


def get_output_result_tables():
    return get_config("output", "result_tables")


def get_output_images():
    return get_config("output", "images")


def get_timeout_in_min():
    return int(get_config("limits", "soft_timeout_in_min"))


def get_hard_timeout_in_min():
    return int(get_config("limits", "hard_timeout_in_min"))


def get_preprocess_data():
    return {k: v.lower() in ['true', '1', 'yes'] for k, v in local_config.items('to_preprocess')}


def get_limits_bool():
    return local_config.getboolean("defaults", "use_limits")


def get_new_runs_bool():
    return local_config.getboolean("defaults", "only_new_evaluations")
