import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

class MFRetrievalModel(tfrs.models.Model):

    def __init__(self, user_model, item_model, task, items):
        super().__init__()
        self.item_model: tf.keras.Model = item_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task
        self.items = items

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the item features and pass them into the item model,
        # getting embeddings back.
        positive_item_embeddings = self.item_model(features["item_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_item_embeddings, compute_metrics=False)

    def predict_for_user(self, user, seen_items_tensor, k=10, shuffle=True):
        # Generate prediction
        if tf.equal(tf.size(seen_items_tensor), 0):
            unseen_items = self.items
        else:
            def_value = tf.constant(-1)
            seen_table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=seen_items_tensor,
                    values=tf.zeros_like(seen_items_tensor, dtype=tf.int32),
                ),
                default_value=def_value
            )
            unseen_items = self.items.filter(lambda x: seen_table[x] == tf.constant(-1)) # We only keep those that are not in the table
        
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends items out of the entire items dataset.
        index.index_from_dataset(
            tf.data.Dataset.zip((unseen_items.batch(100), unseen_items.batch(100).map(self.item_model)))
        )

        # Get recommendations.
        _, titles = index(tf.expand_dims(user, axis=0))
        result = titles[:, :k]
        if shuffle:
            return tf.transpose(tf.random.shuffle(tf.transpose(result)))
        return result

    def predict_all_unseen(self, user, seen_items_tensor, n_items):
        # Generate prediction
        # Skip filtering if there is nothing to be filtered
        if tf.equal(tf.size(seen_items_tensor), 0):
            unseen_items = self.items
        else:
            def_value = tf.constant(-1)
            seen_table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=seen_items_tensor,
                    values=tf.zeros_like(seen_items_tensor, dtype=tf.int32),
                ),
                default_value=def_value
            )
            unseen_items = self.items.filter(lambda x: seen_table[x] == tf.constant(-1)) # We only keep those that are not in the table

        k = n_items - seen_items_tensor.shape[0]

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.user_model, k=k)
        # recommends items out of the entire items dataset.
        index.index_from_dataset(
            tf.data.Dataset.zip((unseen_items.batch(100), unseen_items.batch(100).map(self.item_model)))
        )

        # Get recommendations.
        scores, titles = index(tf.expand_dims(user, axis=0))

        return scores[:, :k], titles[:, :k]

def get_model_mf(unique_user_ids, unique_item_titles, items, embedding_dimension = 32, learning_rate = 0.1):
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_item_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=items.batch(128).map(item_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics,
        #batch_metrics=[tfr.keras.metrics.NDCGMetric()]
    )

    model = MFRetrievalModel(user_model, item_model, task, items)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate))

    return model


