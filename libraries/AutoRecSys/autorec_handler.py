from benchmark_framework.model_base import Model


def load_autorec_and_all_models():
    import tensorflow as tf
    from autorecsys.auto_search import Search
    from autorecsys.pipeline import Input, LatentFactorMapper, RatingPredictionOptimizer, HyperInteraction
    from autorecsys.recommender import RPRecommender
    from sklearn.model_selection import train_test_split

    # From AutoRec's preprocessor base, makes all categorical columns tensorflow floats to be a feasible input
    # Could also be done using pandas or other methods, but reuse to stay as close as possible to the library
    def transform_categorical(data_df, categorical_columns, categorical_filter=3):
        # Step 1: Count categorical occurrences for each column.
        fit_dict = {col: data_df[col].value_counts().to_dict() for col in categorical_columns}

        # Step 2: Reindex categories for each column (create fit dictionary)
        for col, count_dict in fit_dict.items():
            index = 0.0  # float meets TensorFlow type requirement
            categories = list()
            for category, count in count_dict.items():
                if count > categorical_filter:
                    fit_dict[col][category] = index
                    index += 1
                else:
                    categories.append(category)
            for category in categories:
                fit_dict[col][category] = index

        # Step 3: Transform categorical data (apply fit dictionary)
        for col in categorical_columns:
            data_df[col] = data_df[col].map(fit_dict[col])

    class AutoRecModel(Model):
        # Sadly not very Auto as a lot of options still open to chose by user

        def __init__(self):
            super().__init__("AutoRec", None, "AutoRecSys")

        def train(self, dataset):
            x_train, y_train = dataset.train_data
            # 0.1 from default value of autorec's preprocessor
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
            user_num, item_num = dataset.recsys_properties.get_num_values()

            # Only give userid, itemid and rating as input here just like in autosurprise
            # (and as more features are not support by default (not part of docu code / latenfactormappers)
            x_train = x_train[[dataset.recsys_properties.userId_col, dataset.recsys_properties.itemId_col]]
            x_val = x_val[[dataset.recsys_properties.userId_col, dataset.recsys_properties.itemId_col]]
            transform_categorical(x_train, list(x_train))
            transform_categorical(x_val, list(x_val))

            # Setup mappers
            input = Input(shape=[2])
            user_emb = LatentFactorMapper(column_id=0,
                                          num_of_entities=user_num)(input)
            item_emb = LatentFactorMapper(column_id=1,
                                          num_of_entities=item_num)(input)

            # How to choose this? also has no "real" default values... hence keep tutorial code
            output1 = HyperInteraction()([user_emb, item_emb])
            output2 = HyperInteraction()([output1, user_emb, item_emb])
            output3 = HyperInteraction()([output1, output2, user_emb, item_emb])
            output4 = HyperInteraction()([output1, output2, output3, user_emb, item_emb])
            output = RatingPredictionOptimizer()(output4)
            model = RPRecommender(inputs=input, outputs=output)

            # no time limit on searcher, no default max trail number, hence set to 50 for now
            searcher = Search(model=model, tuner_params={'max_trials': 50, 'overwrite': True})
            searcher.search(x=[x_train.values],
                            y=y_train,
                            x_val=[x_val.values],
                            y_val=y_val,
                            # Add early stopping and epochs=100 as default is just non existing (which would be very bad
                            epochs=100,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, min_delta=0.0001)])

            self.model_object = searcher

        def predict(self, dataset):
            x_test, _ = dataset.test_data

            # Transform data similar to transformation in training
            x_test = x_test[[dataset.recsys_properties.userId_col, dataset.recsys_properties.itemId_col]]
            transform_categorical(x_test, list(x_test))

            return self.model_object.predict(x_test)

    return [AutoRecModel]
