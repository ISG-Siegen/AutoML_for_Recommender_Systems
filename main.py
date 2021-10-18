# Main Script Example - can be used to start various subcomponents of the project, see for example the eval part
from utils import lcer
from evaluation import evaler

# Get Logger and Required Config Values
logger = lcer.get_logger("Main")


# Code Section
def placeholder_start():
    evaler.eval_overall_results()


if __name__ == "__main__":
    placeholder_start()
