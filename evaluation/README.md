# Evaluation Scripts

## Script for Dataset Statistics

The `data_statistics.py` script can be used to generate the table (dataframe) with the dataset statistics.

## Script for Results

The `evaler.py` generates plots, tables, and other illustrations for evaluation used in the related paper of this
project. It uses the `overall_benchmark_results.csv`.

## Run Overhead Management

`run_overhead_mgmt.py` is a script that manages which algorithms have been run on which datasets already based on data
in `overall_benchmark_results.csv`. It is used by our `run_comparison.py`script. However, one could also use it
individually to get the current state based on the data. 