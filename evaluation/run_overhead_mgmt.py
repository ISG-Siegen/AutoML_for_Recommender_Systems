# A script to find some data about so far successful runs and data we still need
import os
import sys

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Required Imports
from libraries.name_lib_mapping import NAME_LIB_MAP
from data_processing.preprocessing.data_handler import get_all_dataset_names
from general_utils.lcer import get_logger, get_output_result_data, get_base_path, get_default_metric
from general_utils import filer
from benchmark_framework.dataset_base import DummyDataset
import pandas as pd

# Vars
lib_name = str(sys.argv[1])
algo_names_list = []
fresh_start = False
print_versions = False
logger = get_logger("OverheadMgmt")
DATASET_NAMES = get_all_dataset_names()

# ------------- File management
output_filepath = os.path.join(get_base_path(), get_output_result_data(), "run_overhead_data.json")
output_filepath_benchmark_data = os.path.join(get_base_path(), get_output_result_data(),
                                              "overall_benchmark_results.csv")

# ---- Create overall results output file if needed
if fresh_start and os.path.isfile(output_filepath_benchmark_data):
    os.remove(output_filepath_benchmark_data)

if not os.path.isfile(output_filepath_benchmark_data) or fresh_start:
    filer.write_data(
        pd.DataFrame([], columns=["Dataset", "Model", "LibraryCategory", get_default_metric().name,
                                  "TimeInSeconds", "timestamp", "failed", "fail_reason"]),
        output_filepath_benchmark_data)

so_far_benchmark_data = filer.read_data(output_filepath_benchmark_data)

# ---- Create run overhead data file if needed
if fresh_start and os.path.isfile(output_filepath):
    os.remove(output_filepath)

if not os.path.isfile(output_filepath) or fresh_start:
    filer.write_data_json({"used_dataset_names": DATASET_NAMES, "used_libraries_names": list(NAME_LIB_MAP.keys())},
                          output_filepath)

# Build a dict containing info on run statistics
stats_dict_for_lib = {}

# ------------- Print Used Library Version (if enabled)
if print_versions:
    # print idea from https://stackoverflow.com/a/51056435 (works since we use python3.9 for our docker containers)
    print("#### Version of all used libraries for this framework ####")
    from pip._internal.operations.freeze import freeze

    for requirement in freeze(local_only=True):
        print(requirement)

# ------------- Get all names of algorithms for this library
for algorithm in NAME_LIB_MAP[lib_name]():
    class_name = str(algorithm)

    if hasattr(algorithm, 'requires_dataset'):
        model = algorithm(DummyDataset())
    else:
        model = algorithm()

    model_name = model.name
    algo_names_list.append(model_name)
    del model

# ------------- Get the names of each model that still needs to be run for a dataset
for dataset_name in DATASET_NAMES:
    tmp_df = so_far_benchmark_data[so_far_benchmark_data["Dataset"] == dataset_name]
    algos_run_so_far_on_dataset = set(tmp_df["Model"].unique())
    algos_to_run_on_dataset = set(algo_names_list) - algos_run_so_far_on_dataset
    # Add data to dict
    stats_dict_for_lib[dataset_name] = list(algos_to_run_on_dataset)

stats_dict_for_lib["model_names"] = algo_names_list
stats_dict_for_lib["all_benchmarks_done_at_least_once"] = not any([bool(stats_dict_for_lib[x]) for x in DATASET_NAMES])

# Read dict, update dict, write dict
current_data = filer.read_data_json(output_filepath)
# Update names to make sure correct state is maintained
current_data.update({"used_dataset_names": DATASET_NAMES, "used_libraries_names": list(NAME_LIB_MAP.keys())})
current_data.update({lib_name: stats_dict_for_lib})
filer.write_data_json(current_data, output_filepath)
