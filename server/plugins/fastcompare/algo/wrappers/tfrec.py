from abc import ABC
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType
from plugins.utils.tfrs_model import get_model_25m
from plugins.utils.preference_elicitation import prepare_tf_data

import tensorflow as tf

class TFRecommendersWrapper(AlgorithmBase, ABC):
    def __init__(self, loader, model, epochs, **kwargs):
        self.model = model
        self.epochs = epochs
        self.loader = loader
        # data is pandas dataset, we have to transform it for tensorflow
        ratings_df = self.loader.ratings_df.copy()
        # Add movie_title
        ratings_df.loc[:, "movie_title"] = ratings_df.movieId.map(self.loader.items_df_indexed.title)
        # Rename column and cast to string
        ratings_df = ratings_df.rename(columns={"userId": "user_id"})
        self.new_user = str(ratings_df.user_id.max() + 1)
        ratings_df.user_id = ratings_df.user_id.astype(str)
        ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[["user_id", "movie_title"]]))
        train = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=True)
        self.train = train # Store train for future use
        self.dataset = train.batch(8192).cache()
        self.movies = tf.data.Dataset.from_tensor_slices(dict(self.loader.movies_df.rename(columns={"title": "movie_title"})[["movie_title"]])).map(lambda x: x["movie_title"])

    # One-time fitting of the model for a predefined number of epochs
    def fit(self):
        self.model.fit(self.dataset, epochs=self.epochs)


    # Predict for the user, we can also pass-in ratings to update user's preferences (a sort of fine-tune)
    def predict(self, selected_items, filter_out_items, k):
        new_user = tf.constant(self.new_user) # Create tensor for the new user
        # Generate new "dataset" based on user's selected items
        def data_gen():
            for x in selected_items:
                yield {
                    "movie_title": tf.constant(self.loader.movies_df.loc[x].title),
                    "user_id": new_user,
                }
        ratings2 = tf.data.Dataset.from_generator(data_gen, output_signature={
            "movie_title": tf.TensorSpec(shape=(), dtype=tf.string),
            "user_id": tf.TensorSpec(shape=(), dtype=tf.string)
        })

        # Finetune the model (we add few more samples from self.train to increase stability)
        # fine-tuning for two epochs seems fine
        self.model.fit(ratings2.concatenate(self.train.take(100)).batch(256), epochs=2)

        seen_movies_tensor = tf.stack(
            [tf.constant(self.loader.movies_df.loc[x].title) for x in selected_items]
            +
            [tf.constant(self.loader.movies_df.loc[x].title) for x in filter_out_items]
        )
        
        predictions = tf.squeeze(self.model.predict_for_user(new_user, seen_movies_tensor, k)).numpy()
        
        # The underlying model has to implement predict_for_user since it is not obvious how to do this in a generic way in TFRecommenders (models may behave very differently so just having generic predict() method is not enough because it is not obvious whetehr it will return scores, items or anything else)
        top_k = [self.loader.movie_id_to_index[self.loader.movies_df[self.loader.movies_df.title == x.decode("UTF-8")].movieId.values[0]] for x in predictions]

        return top_k

    
    # Custom saving and loading is needed for tensorflow
    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


class SimpleMatrixFactorization(TFRecommendersWrapper):
    def __init__(self, loader, epochs, embedding_dimension, learning_rate, **kwargs):
        # here we use framework's common functionality and further wrap it
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate
        self.loader = loader
        self.unique_user_ids, self.unique_movie_titles, self.movies, _, _ = prepare_tf_data(self.loader)
        model = get_model_25m(self.unique_user_ids, self.unique_movie_titles, self.movies, self.embedding_dimension, self.learning_rate)
        super().__init__(loader, model, epochs, **kwargs)

    @classmethod
    def name(cls):
        return "TF Recommenders Matrix Factorization"

    @classmethod
    def parameters(cls):
        return [
            Parameter("epochs", ParameterType.INT, 5, help_key="simple_mf_epochs_help"),
            Parameter("embedding_dimension", ParameterType.INT, 32, help_key="simple_mf_embedding_dim_help"),
            Parameter("learning_rate", ParameterType.FLOAT, 0.1, help_key="simple_mf_lr_help"),
        ]