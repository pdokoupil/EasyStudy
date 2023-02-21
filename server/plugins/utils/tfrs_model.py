import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

class MovielensRetrievalModel(tfrs.models.Model):

    def __init__(self, user_model, movie_model, task, movies):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task
        self.movies = movies

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=False)

    def predict_for_user(self, user, seen_movies_tensor, k=10, shuffle=True):
        # Generate prediction
        if tf.equal(tf.size(seen_movies_tensor), 0):
            unseen_movies = self.movies
        else:
            def_value = tf.constant(-1)
            seen_table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=seen_movies_tensor,
                    values=tf.zeros_like(seen_movies_tensor, dtype=tf.int32),
                ),
                default_value=def_value
            )
            unseen_movies = self.movies.filter(lambda x: seen_table[x] == tf.constant(-1)) # We only keep those that are not in the table
        
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            #tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
            tf.data.Dataset.zip((unseen_movies.batch(100), unseen_movies.batch(100).map(self.movie_model)))
        )

        # Get recommendations.
        _, titles = index(tf.expand_dims(user, axis=0))
        result = titles[:, :k]
        if shuffle:
            return tf.transpose(tf.random.shuffle(tf.transpose(result)))
        return result

    def predict_all_unseen(self, user, seen_movies_tensor, n_items):
        # Generate prediction
        # Skip filtering if there is nothing to be filtered
        if tf.equal(tf.size(seen_movies_tensor), 0):
            unseen_movies = self.movies
        else:
            def_value = tf.constant(-1)
            seen_table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=seen_movies_tensor,
                    values=tf.zeros_like(seen_movies_tensor, dtype=tf.int32),
                ),
                default_value=def_value
            )
            unseen_movies = self.movies.filter(lambda x: seen_table[x] == tf.constant(-1)) # We only keep those that are not in the table

        k = n_items - seen_movies_tensor.shape[0]

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            #tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
            tf.data.Dataset.zip((unseen_movies.batch(100), unseen_movies.batch(100).map(self.movie_model)))
        )

        # Get recommendations.
        scores, titles = index(tf.expand_dims(user, axis=0))

        return scores[:, :k], titles[:, :k]

    # User's unseen movies
    def _users_unseen_movies(self, movies, users_seen_movies):
        def fnc_impl(x):
            return x.numpy() not in users_seen_movies
        def fnc(x):
            return tf.py_function(fnc_impl, [x], tf.bool)
        return movies.filter(fnc)

    def _filter_user(self, x, user):
        return x["user_id"] != user

def get_model_25m(unique_user_ids, unique_movie_titles, movies, embedding_dimension = 32, learning_rate = 0.1):
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics,
        #batch_metrics=[tfr.keras.metrics.NDCGMetric()]
    )

    model = MovielensRetrievalModel(user_model, movie_model, task, movies)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate))

    return model


