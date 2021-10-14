from benchmark_framework import benchmarker, metrics, dataset_base
from data_processing.preprocessing.preprocessing_100k import load_ml_100k
from libraries.scikit_learn.random_forest import RF

# Get data
data, features, label = load_ml_100k()
dataset = dataset_base.Dataset("movielens-100k", data, features, label)

# Setup benchmark for dataset ml100k and the model RF
example_benchmark = benchmarker.Benchmark(dataset, metrics.RSME(), 60, RF())

# Execute
metric_val, execution_time = example_benchmark.run()

print("{}: RSME of {} | Time take {}".format(example_benchmark.model.name, metric_val, execution_time))
