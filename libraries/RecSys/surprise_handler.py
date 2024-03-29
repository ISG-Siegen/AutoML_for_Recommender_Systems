from benchmark_framework.model_base import Model
from abc import abstractmethod


def load_surprise_and_all_models():
    from surprise import SVD, SVDpp
    from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
    from surprise import CoClustering, BaselineOnly, SlopeOne, NMF, NormalPredictor
    from general_utils.surprise_utils import predict
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

            # Fix Problems with NMF not being able to handle zero ratings
            if self.name == "Surprise_NMF":
                # To do so, make 0 values to epsilon
                p_x_train.loc[p_x_train[label[0]] == 0, label] = 0.0000001

            # Adapt custom Dataframe to Surprise Lib requirements
            reader = Reader(rating_scale=(dataset.recsys_properties.rating_lower_bound,
                                          dataset.recsys_properties.rating_upper_bound))

            # Surprise only supports userId, itemID, rating, timestamp as input but no additional features
            # However, no of the current and used out-of-the-box implementations uses the timestamp
            # Furthermore, the timestamp is also only supported when loading from a file.
            # Consequently, only userid, itemid, rating is used by all surprise algorithms
            # See for issues talking about this:
            # https://github.com/NicolasHug/Surprise/issues/85; https://github.com/NicolasHug/Surprise/issues/314

            # Load dataset with relevant columns and reader
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
    surprise_models = [SurpriseSVD, SurpriseSVDpp, SurpriseCoClustering, SurpriseBaselineOnly,
                       SurpriseNMF, SurpriseNormalPredictor]
    memory_allocation_problematic_models = [SurpriseKNNWithMeans, SurpriseKNNBaseline, SurpriseKNNBasic,
                                            SurpriseKNNWithZScore, SurpriseSlopeOne]

    return surprise_models + memory_allocation_problematic_models
