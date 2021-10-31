from benchmark_framework.model_base import Model


def load_autorec_and_all_models():
    import tensorflow as tf
    from autorec.auto_search import Search
    from autorec.pipeline import Input, LatentFactorMapper, RatingPredictionOptimizer, ElementwiseInteraction
    from autorec.pipeline.preprocessor import MovielensPreprocessor, NetflixPrizePreprocessor
    from autorec.recommender import RPRecommender


    # TODO https://github.com/datamllab/AutoRec

    class AutoRecModel(Model):
        def __init__(self):
            super().__init__("AutoRec", None, "AutoRecSys")

        def train(self, dataset):
            movielens = MovielensPreprocessor()
            train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()
            train_X_categorical = movielens.get_x_categorical(train_X)
            val_X_categorical = movielens.get_x_categorical(val_X)
            test_X_categorical = movielens.get_x_categorical(test_X)
            user_num, item_num = movielens.get_hash_size()

        def predict(self, dataset):

            return predictions

    return []
