# Raw Error Result Data

## Info

* `run_overhead_data.json` contains information used to manage which evaluations still have to be run
* `overall_benchmark_results.csv` contains the error of an algorithm for a dataset per row

## Remarks

* The entry for `yelp` and `TPOT_Regressor` was added manually, since our tool was not able to capture the memout error
  by itself. The TPOT_Regressor did catch the memory error itself and terminated unexpectedly. We were not able to
  capture this bug with our implementation.