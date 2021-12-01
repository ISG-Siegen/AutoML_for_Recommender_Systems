# !Make sure you have select the correct environment with appropriate packages that need to be loaded !
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
from data_processing.preprocessing.main_preprocessing import load_ml_100k_from_csv

dataset = dataset_base.Dataset(*load_ml_100k_from_csv())

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
    rsme_score = metrics.RSME().evaluate(dataset, y_pred)
    print("RSME: {}".format(rsme_score))

    print("Done \n")

print("Everything worked")
