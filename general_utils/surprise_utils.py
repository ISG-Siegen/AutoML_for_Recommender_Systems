# A Tool for basic file handling
import pandas as pd

'''
predict method copied from microsoft recommenders
https://github.com/microsoft/recommenders/blob/main/recommenders/models/surprise/surprise_utils.py
'''


def predict(algo, data, usercol,  itemcol, predcol):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pandas.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column
    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    predictions = [
        algo.predict(getattr(row, usercol), getattr(row, itemcol))
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(
        index=str, columns={"uid": usercol, "iid": itemcol, "est": predcol}
    )
    return predictions.drop(["details", "r_ui"], axis="columns")
