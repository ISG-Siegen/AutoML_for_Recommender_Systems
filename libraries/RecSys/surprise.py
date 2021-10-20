from benchmark_framework.model_base import Model
from abc import abstractmethod
from surprise import SVD, KNNBasic
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
class SurpriseSGD(SurpriseModel):

    def __init__(self):
        super().__init__("Surprise_SingularValueDecompositionAlgorithm", SVD())


class SurpriseKNN(SurpriseModel):

    def __init__(self):
        super().__init__("Surprise_KNeighborsNeighbor", KNNBasic())
