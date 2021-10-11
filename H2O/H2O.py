import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import time

#___________________________________________________MODEL_______________________________________________________________
def H2O(pandas_df):
    h2o.init()
    dataFrame = h2o.H2OFrame(pandas_df)
    train, valid = dataFrame.split_frame(ratios=[.75])
    model = H2OGradientBoostingEstimator()
    start = time.time()
    model.train(x = ["movieId", "userId", 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western'], y= "rating", training_frame=train, validation_frame=valid)
    stop = time.time()
    execution_time = stop-start
    rmse = model.rmse()
    return rmse, execution_time






