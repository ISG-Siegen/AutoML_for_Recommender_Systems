from benchmark_framework import benchmarker, metrics, dataset_base
from libraries.scikit_learn.scikit_models import RF, SVRegressor, SGD, KNN
from data_processing.preprocessing.preprocessing_100k import load_ml_100k
from libraries.H2O.H2O import H2OHandler
from libraries.auto_scikit_learn.automl import AutoSKLearn

# Get data
data, features, label = load_ml_100k()
dataset = dataset_base.Dataset("movielens-100k", data, features, label)

# Setup benchmark for dataset ml100k and the model RF

benchmarks = []
for model_base in [RF, SGD, SVRegressor, KNN, H2OHandler, AutoSKLearn]:
    benchmarks.append(benchmarker.Benchmark(dataset, metrics.RSME(), 60, model_base()))

# Execute
result_data = []
for benchmark in benchmarks:
    result_data.append((benchmark.model.name, *benchmark.run()))

# Evaluate
for model_name, metric_val, execution_time in result_data:
    print("{}: RSME of {} | Time take {}".format(model_name, metric_val, execution_time))
