# Main Script Example - can be used to start various subcomponents of the project, see for example the eval part
from utils import lcer
from evaluation import evaler

# Get Logger and Required Config Values
logger = lcer.get_logger("Main")
TIMEOUT = lcer.get_settings_timeoutinmin()
DATA_PATH = lcer.get_dataset_testdata()


# Code Section
def placeholder_start():
    evaler.full_data_eval(DATA_PATH, save_images=False)


if __name__ == "__main__":
    placeholder_start()
