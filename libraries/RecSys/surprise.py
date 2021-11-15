from benchmark_framework.model_base import Model
from abc import abstractmethod


def load_surprise_and_all_models():
    from surprise import SVD, SVDpp
    from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
    from surprise import CoClustering, BaselineOnly, SlopeOne, NMF, NormalPredictor
    from utils.surprise_utils import predict
    from surprise import Dataset, Reader

    # Default interface for Surprise
    class SurpriseModel(Model):

        @abstractmethod
        def __init__(self, name, model):
            super().__init__(name, model, "RecSys")

        def train(self, dataset):
            x_train, y_train = dataset.train_data

            label = list(y_train)

            # Build 1 training frame
            p_x_train = x_train.copy()
            p_x_train[label] = y_train

            # Adapt custom Dataframe to Surprise Lib requirements
            reader = Reader(rating_scale=(dataset.recsys_properties.rating_lower_bound,
                                          dataset.recsys_properties.rating_upper_bound))
            data = Dataset.load_from_df(p_x_train[[dataset.recsys_properties.userId_col,
                                                   dataset.recsys_properties.itemId_col,
                                                   dataset.recsys_properties.rating_col]], reader)
            trainset = data.build_full_trainset()
            self.model_object.fit(trainset)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            predictions = predict(self.model_object, x_test,
                                  usercol=dataset.recsys_properties.userId_col,
                                  itemcol=dataset.recsys_properties.itemId_col,
                                  predcol='prediction')
            return predictions['prediction']

    # Different Models from the library
    class SurpriseSVD(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_SingularValueDecompositionAlgorithm", SVD())

    class SurpriseKNNBasic(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_KNNBasic", KNNBasic())

    class SurpriseKNNBaseline(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_KNNBaseline", KNNBaseline())

    class SurpriseKNNWithZScore(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_KNNWithZScore", KNNWithZScore())

    class SurpriseKNNWithMeans(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_KNNWithMeans", KNNWithMeans())

    class SurpriseCoClustering(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_CoClustering", CoClustering())

    class SurpriseBaselineOnly(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_BaselineOnly", BaselineOnly())

    class SurpriseSlopeOne(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_SlopeOne", SlopeOne())

    class SurpriseSVDpp(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_SVDpp", SVDpp())

    class SurpriseNMF(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_NMF", NMF())

    class SurpriseNormalPredictor(SurpriseModel):

        def __init__(self):
            super().__init__("Surprise_NormalPredictor", NormalPredictor())

    # Return
    surprise_models = [SurpriseSVD, SurpriseSVDpp, SurpriseKNNBasic, SurpriseKNNBaseline, SurpriseKNNWithZScore,
                       SurpriseKNNWithMeans, SurpriseCoClustering, SurpriseBaselineOnly, SurpriseSlopeOne,
                       SurpriseNMF, SurpriseNormalPredictor]
    return surprise_models
