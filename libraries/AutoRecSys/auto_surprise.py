from benchmark_framework.model_base import Model


def load_auto_surprise_and_all_models():
    from surprise import Dataset, Reader
    from auto_surprise.engine import Engine
    from utils.surprise_utils import predict

    class AutoSurpriseModel(Model):
        def __init__(self):
            super().__init__("AutoSurprise", None, "AutoRecSys")

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

            engine = Engine(verbose=False)
            best_algo, best_params, best_score, tasks = engine.train(
                data=data,
                target_metric='test_rmse',
                cpu_time_limit=60 * 60
            )
            # create best model object
            self.model_object = engine.build_model(best_algo, best_params)

            # train created model
            trainset = data.build_full_trainset()
            self.model_object.fit(trainset)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            predictions = predict(self.model_object, x_test,
                                  usercol=dataset.recsys_properties.userId_col,
                                  itemcol=dataset.recsys_properties.itemId_col,
                                  predcol='prediction')
            return predictions['prediction']

    return [AutoSurpriseModel]
