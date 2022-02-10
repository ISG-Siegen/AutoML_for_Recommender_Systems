# !Make sure you have select the correct environment with appropriate packages that need to be loaded !
# !Make sure you have the dataset that is used here on your system (a .csv and .json file) - ask lennart or tobias
# [Ignore This] ------------- Ensure that base path is found
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

# [CHANGE THIS] -------- Example Test for lenskit
from libraries.RecSys.lenskit_handler import load_lenskit_and_all_models

lenskit_models = load_lenskit_and_all_models()
loaded_models_to_test = lenskit_models  # Set the name for the test case below

# [NO NEED TO CHANGE THIS] ---------------- Test if implementation works
# Get a test dataset (here ml 100k)
from benchmark_framework import dataset_base, metrics
from data_processing.preprocessing.data_handler import load_datasets_information, load_from_files

data_for_load_dataset = [x for x in load_datasets_information() if x[2] == "movielens-100K"][0]  # use ml100k dataset
dataset = dataset_base.Dataset(*load_from_files(data_for_load_dataset[0], data_for_load_dataset[1]))

for model in loaded_models_to_test:

    # Init Model
    print("Init Model")
    if hasattr(model, 'requires_dataset'):
        model = model(dataset)
    else:
        model = model()
    print("Test Model {}".format(model.name))
    # Fit / train
    print("Fit")
    model.train(dataset)
    # Predict
    print("Predict")
    y_pred = model.predict(dataset)
    # Score
    rmse_score = metrics.RMSE().evaluate(dataset, y_pred)
    print("RMSE: {}".format(rmse_score))

    print("Done \n")

print("Everything worked")
