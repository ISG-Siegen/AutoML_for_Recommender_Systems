from benchmark_framework.model_base import Model


def load_ktrain_and_all_models():
    import ktrain
    from ktrain import tabular

    # Note: ktrain is very much WIP and thus the "default" values are a bit problematic
    # Furthermore, it could be understood as Semi-AutoML, but for it counts as ML

    class KtrainModel(Model):

        def __init__(self):
            super().__init__("ktrainRegressor", None, "ML")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            train_df = x_train.copy()
            train_df["label_col"] = y_train

            trn, val, preproc = tabular.tabular_from_df(train_df, is_regression=True, label_columns='label_col')
            model = tabular.tabular_regression_model('mlp', trn)
            learner = ktrain.get_learner(model, train_data=trn, val_data=val)
            # learner.lr_find()  does not automatically return anything
            learner.autofit(1e-3)  # default to 0.001 for now
            self.model_object = ktrain.get_predictor(model, preproc)

        def predict(self, dataset):
            x_test, _ = dataset.test_data
            return self.model_object.predict(x_test)

    return [KtrainModel]
