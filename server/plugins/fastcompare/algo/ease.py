from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class EASE(AlgorithmBase, ABC):
    """Implementation of EASE algorithm for EasyStudy using closed-form solution
    paper: https://arxiv.org/abs/1905.03375


    Internally, we assign binary ratings during training as that is the way it
    was done in the paper. As an alternative, we could predict the ratings, as
    is also mentioned in the paper.
    """

    def __init__(self, loader, positive_threshold, l2, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._threshold = positive_threshold
        self._l2 = l2

        self._items_count = np.shape(self._rating_matrix)[1]

        self._weights = None

    # One-time fitting of the algorithm for a predefined number of iterations
    def fit(self):
        X = tf.convert_to_tensor(
            np.where(self._rating_matrix >= self._threshold, 1, 0), dtype=tf.float32
        )
        G = tf.transpose(X) @ X
        G += self._l2 * tf.linalg.tensor_diag([1.0 for _ in range(self._items_count)])

        P = tf.linalg.inv(G)

        B = P / (-tf.linalg.tensor_diag_part(P))
        B = tf.linalg.set_diag(B, tf.zeros(B.shape[0]))
        self._weights = B

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        rat = pd.DataFrame({"item": selected_items}).set_index("item", drop=False)
        # Appropriately filter out what was seen and what else should be filtered
        candidates = np.setdiff1d(self._all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)
        if not selected_items:
            # Nothing was selected, since the new_user was unknown during training, Lenskit algorithm would simply recommended nothing
            # to avoid empty recommendation, we just sample random candidates
            return np.random.choice(candidates, size=k, replace=False).tolist()
        indices = list(selected_items)
        user_vector = np.zeros((self._items_count,))
        for i in indices:
            user_vector[i] = 1.0

        preds = tf.tensordot(
            tf.convert_to_tensor(user_vector, dtype=tf.float32), self._weights, 1
        ).numpy()

        candidates_by_prob = sorted(
            ((preds[cand], cand) for cand in candidates), reverse=True
        )
        result = [x for _, x in candidates_by_prob][:k]

        return result

    @classmethod
    def name(cls):
        return "EASE"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "l2",
                ParameterType.FLOAT,
                0.1,  # I did not find a value in the paper, we can try tweaking the default value in the future
                help="L2-norm regularization",
                help_key="ease_l2_help",
            ),
            Parameter(  # at the moment, we assume that greater ratings are better
                "positive_threshold",
                ParameterType.FLOAT,
                2.5,
                help="Threshold for conversion of n-ary rating into binary (positive/negative).",
            ),
        ]
