from benchmark_framework.model_base import Model
from abc import abstractmethod


def load_lenskit_and_all_models():
    # Disable logging overhead from lenskit
    import logging
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("lenskit").setLevel(logging.WARNING)

    from lenskit.algorithms import Recommender

    from lenskit.algorithms.basic import Fallback, Bias

    from lenskit.algorithms.user_knn import UserUser
    from lenskit.algorithms.item_knn import ItemItem

    from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
    from lenskit.algorithms.funksvd import FunkSVD

    from lenskit.algorithms.tf import BiasedMF as TFBiasedMF, BPR, IntegratedBiasMF

    from lenskit.algorithms.hpf import HPF

    from lenskit.algorithms.svd import BiasedSVD

    class LenskitModel(Model):
        no_neighbours = 20  # set to 20 as no default value exist but this is commonly used in tutorials from lenskit
        no_features = 3  # TODO more features?

        @abstractmethod
        def __init__(self, name, model):
            super().__init__('lenskit_' + name, model, "RecSys")
            self.base = Bias()

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            label = list(y_train)

            # Build 1 training frame
            p_x_train = x_train.copy()
            p_x_train[label] = y_train

            # TODO more features?
            p_x_train = p_x_train[[dataset.recsys_properties.userId_col, dataset.recsys_properties.itemId_col,
                                   dataset.recsys_properties.rating_col]]
            # Rename important RecSys columns which are needed by lenskit
            p_x_train = p_x_train.rename(
                columns={dataset.recsys_properties.userId_col: "user", dataset.recsys_properties.itemId_col: "item",
                         dataset.recsys_properties.rating_col: "rating"})

            # ensure that the algorithm is a class Recommender,
            self.model_object = Recommender.adapt(self.model_object)
            self.base = Recommender.adapt(self.base)

            # Train
            self.model_object.fit(p_x_train)
            self.base.fit(p_x_train)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            p_x_test = x_test.copy()

            p_x_test = p_x_test.rename(
                columns={dataset.recsys_properties.userId_col: "user", dataset.recsys_properties.itemId_col: "item"})
            data = p_x_test[["user", "item"]]

            predictor = Fallback(self.model_object, self.base)  # Use bias as base following documentation

            return predictor.predict(data)

    # Different Models from the library
    # -- Knn algorithms
    class ItemItemKNN(LenskitModel):

        def __init__(self):
            super().__init__("ItemItem", ItemItem(self.no_neighbours))

    class UserUserKNN(LenskitModel):

        def __init__(self):
            super().__init__("UserUser", UserUser(self.no_neighbours))

    # -- Basic Algorithms
    class BiasAlgorithm(LenskitModel):
        def __init__(self):
            super().__init__("Bias", Bias())

    # -- matrix factorization
    class ALSBiasedMFAlgorithm(LenskitModel):
        def __init__(self):
            super().__init__("ALSBiasedMF", ALSBiasedMF(self.no_features))

    class SVD_Funk(LenskitModel):
        def __init__(self):
            super().__init__("FunkSVD", FunkSVD(self.no_features))

    class SVD_Biased(LenskitModel):
        def __init__(self):
            super().__init__("BiasedSVD", BiasedSVD(self.no_features))

    # Tensorflow
    class TFBiasedMFAlgorithm(LenskitModel):
        def __init__(self):
            super().__init__("TFBiasedMF", TFBiasedMF(self.no_features))

    class BPRAlgorithm(LenskitModel):
        def __init__(self):
            super().__init__("BPR", BPR(self.no_features))

    class IntegratedBiasMFAlgorithm(LenskitModel):
        def __init__(self):
            super().__init__("IntegratedBiasMF", IntegratedBiasMF(self.no_features))

    # External Libraries
    class PredictorHPF(LenskitModel):
        def __init__(self):
            super().__init__("HPF", HPF(self.no_features))

    # Return
    lenskit_model = [ItemItemKNN, UserUserKNN,
                     BiasAlgorithm,
                     ALSBiasedMFAlgorithm, SVD_Funk, SVD_Biased, PredictorHPF,
                     IntegratedBiasMFAlgorithm, BPRAlgorithm, TFBiasedMFAlgorithm]

    # -- Notes on missing algorithms
    # Do not use Popular as it only can recommend not predict
    # Do not use implicit MF as it is for implicit (similar implicit.BPR implicit.ALS)

    return lenskit_model
