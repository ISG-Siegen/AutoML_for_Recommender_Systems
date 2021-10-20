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

    def train(self, x_train, y_train):
        label = list(y_train)

        # Build 1 training frame
        p_x_train = x_train.copy()
        p_x_train[label] = y_train

        # Adapt custom Dataframe to Surprise Lib requirements
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(p_x_train[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        self.model_object.fit(trainset)

    def predict(self, x_test):
        predictions = predict(self.model_object, x_test, usercol='userId', itemcol='movieId', predcol='prediction')
        predictions.head()
        return predictions['prediction']


# Different Models from the library
class SurpriseKNN(SurpriseModel):

    def __init__(self):
        super().__init__("SingularValueDecompositionAlgorithm", SVD())


class SurpriseSGD(SurpriseModel):

    def __init__(self):
        super().__init__("KNeighborsNeighbor", KNNBasic())