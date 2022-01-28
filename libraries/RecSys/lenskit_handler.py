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

            p_x_train = p_x_train.rename(
                columns={dataset.recsys_properties.userId_col: "user", dataset.recsys_properties.itemId_col: "item",
                         dataset.recsys_properties.rating_col: "rating",
                         dataset.recsys_properties.timestamp_col: "timestamp"})

            # ensure that the algorithm is a class Recommender,
            self.model_object = Recommender.adapt(self.model_object)
            self.base = Recommender.adapt(self.base)

            # All of our used lenskit algorithms do not use Timestamp, thus as a result duplicates exit (e.g. for edits or repeated reviews).
            # These duplicates are however not handled by lenskit and crashes these algorithms during prediction.
            # To solve this, drop duplicates here.
            # Note: this changes the training set compared to other algos. However, without this, training would not work at all
            # This is an error that should be handled internally by lenskit but is not...
            p_x_train = p_x_train[~p_x_train[["user", "item"]].duplicated()]

            # Train
            self.model_object.fit(p_x_train)
            self.base.fit(p_x_train)

        def predict(self, dataset):
            """Lenskit's predict function becomes very complicated because we need to handle duplicate errors."""
            x_test, _ = dataset.test_data
            p_x_test = x_test.copy()

            p_x_test = p_x_test.rename(
                columns={dataset.recsys_properties.userId_col: "user", dataset.recsys_properties.itemId_col: "item",
                         dataset.recsys_properties.timestamp_col: "timestamp"})
            predictor = Fallback(self.model_object, self.base)  # Use bias as base following documentation

            # Remove duplicates here to avoid breaking prediction (will be filled later on with predicted value)
            dp_mask = p_x_test[["user", "item"]].duplicated()
            d_p_x_test = p_x_test[~dp_mask]

            # Get predictions for subset
            predictions = predictor.predict(d_p_x_test)

            # If we remove duplicates in training and predict (potentially) with duplicates here,
            # lenskit will for some reason return an array with more predictions than given inputs in some cases.
            # That is, for some of the input predictions, multiple predictions are made.
            # All theses inputs are itself user,item duplicates in the test dataset.
            # In short, lenskit produces duplicate predictions for each duplicate input.
            # Solution: Remove duplicate predictions (only use first occurrence)
            predictions = predictions[~predictions.index.duplicated()]

            # Re-order predictions
            predictions = predictions.reindex(p_x_test.index)

            # predictions will contain nan for each index that was a duplicate if above any duplicates were removed
            # These must be filled now by finding the prediction values for all duplicates.
            # Re-using the prediction is not possible as the duplicates can also have duplicates.
            for index, row in p_x_test[dp_mask][["user", "item"]].iterrows():
                # Get index of original value
                t = d_p_x_test.index[(d_p_x_test["user"] == row["user"])
                                     & (d_p_x_test["item"] == row["item"])].values[0]
                # Get Prediction
                pred_val = predictions[t]
                # Fill prediction
                predictions[index] = pred_val

            # Verify correctness of return value
            try:
                assert predictions.index.tolist() == p_x_test.index.tolist()
            except AssertionError:
                raise AssertionError("Predictions and Test data do not have the same index list." +
                                     "Returned predictions are likely incorrect ordered or otherwise broken.")

            return predictions

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
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("ALSBiasedMF", ALSBiasedMF(len(dataset.features)))

    class SVD_Funk(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("FunkSVD", FunkSVD(len(dataset.features)))

    class SVD_Biased(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("BiasedSVD", BiasedSVD(len(dataset.features)))

    # Tensorflow
    class TFBiasedMFAlgorithm(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("TFBiasedMF", TFBiasedMF(len(dataset.features)))

    class BPRAlgorithm(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("BPR", BPR(len(dataset.features)))

    class IntegratedBiasMFAlgorithm(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("IntegratedBiasMF", IntegratedBiasMF(len(dataset.features)))

    # External Libraries
    class PredictorHPF(LenskitModel):
        requires_dataset = True

        def __init__(self, dataset):
            super().__init__("HPF", HPF(len(dataset.features)))

    # Return
    lenskit_model = [ItemItemKNN, UserUserKNN, BiasAlgorithm, ALSBiasedMFAlgorithm, SVD_Funk, SVD_Biased, PredictorHPF,
                     IntegratedBiasMFAlgorithm, BPRAlgorithm, TFBiasedMFAlgorithm]

    # -- Notes on missing algorithms
    # Do not use Popular as it only can recommend not predict
    # Do not use implicit MF as it is for implicit (similar implicit.BPR implicit.ALS)

    return lenskit_model
