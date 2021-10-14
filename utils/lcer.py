# A Tool for handling logging and configurations
import logging
from configparser import ConfigParser
import os

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


# --------------- Config Code
def get_config(category, entry):
    return local_config[category][entry]


def get_dataset_testdata():
    return get_config("datasets", "test_data")


def get_output_images():
    return get_config("output", "images")


def get_settings_timeoutinmin():
    return get_config("settings", "timeout_in_min")
