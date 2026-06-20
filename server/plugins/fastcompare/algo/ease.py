# pyright: reportConstantRedefinition=false

from abc import ABC

import numpy as np
import numpy.typing as npt
import pandas as pd
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)
from plugins.fastcompare.algo.wrappers.data_loadering import MLDataLoaderWrapper


class EASE(AlgorithmBase, ABC):
    """Implementation of EASE algorithm for EasyStudy using closed-form solution
    paper: https://arxiv.org/abs/1905.03375


    Internally, we assign binary ratings during training as that is the way it
    was done in the paper. As an alternative, we could predict the ratings, as
    is also mentioned in the paper.
    """

    # the loader type should be DataLoaderBase, but this makes the developer experience better and is good enough for this branch
    def __init__(self, loader: MLDataLoaderWrapper, positive_threshold: float, l2: float, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items: npt.NDArray[np.int64] = self._ratings_df.item.unique()

        self._rating_matrix: npt.NDArray[np.float32] = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        self._threshold = positive_threshold
        self._l2 = l2

        self._items_count: int = np.shape(self._rating_matrix)[1]

    # One-time fitting of the algorithm for a predefined number of iterations
    def fit(self):
        X = np.where(self._rating_matrix >= self._threshold, 1, 0).astype(np.float32)

        # Compute Gram matrix (G = X^T @ X)
        G = X.T @ X
        G += self._l2 * np.eye(self._items_count)  # Regularization

        # Compute the inverse of G
        P = np.linalg.inv(G)

        # Compute B matrix
        diag_P = np.diag(P)
        B = P / (-diag_P[:, None])  # Normalize rows by diagonal elements
        np.fill_diagonal(B, 0)  # Set diagonal to zero

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
        user_vector = np.zeros((self._items_count,), dtype=np.float32)
        for i in indices:
            user_vector[i] = 1.0

        preds = np.dot(user_vector, self._weights)

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
                500,  # the paper's recommended value for MovieLens-20M
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
