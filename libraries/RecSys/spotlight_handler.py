from benchmark_framework.model_base import Model


def load_spotlight():
    from spotlight.factorization.explicit import ExplicitFactorizationModel
    from spotlight.interactions import Interactions

    # important remarks regarding spotlight:
    # 1- it has only one model, that could be used for explicit feedback case-study
    # 2- ItemID, userID, rating and timestamp could be used as features to create the interactor.
    #    However, timestamp is not used by the algorithms. Thus, it was omitted.

    class SpotlightModel(Model):
        def __init__(self):
            super().__init__("Spotlight", None, "RecSys")

        def train(self, dataset):
            # default parameters
            self.model = ExplicitFactorizationModel()

            x_train, y_train = dataset.train_data
            itemID_train = (x_train['itemId'].values)
            userID_train = (x_train['userId'].values)
            rating_train = (y_train.values.flatten())

            train_interactions = Interactions(userID_train, itemID_train, rating_train)
            self.model.fit(train_interactions, verbose=True)

        def predict(self, dataset):
            x_test, y_test = dataset.test_data
            itemID_test = (x_test['itemId'].values)
            userID_test = (x_test['userId'].values)
            rating_test = (y_test.values.flatten())

            test_interactions = Interactions(userID_test, itemID_test, rating_test)
            predictions = self.model.predict(test_interactions.user_ids, test_interactions.item_ids)

            return predictions

    return [SpotlightModel]
