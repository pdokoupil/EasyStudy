"""Pure-NumPy baseline algorithms that ship in the lightweight core.

The Most-Popular / Random baselines historically lived in ``wrappers/lenskit.py`` and thus
required the ``[lenskit]`` extra. That left the lightweight core with only EASE — too few to
run a fastcompare comparison (which needs 2–3 algorithms). These NumPy reimplementations have
no heavy dependencies, so ``pip install easystudy`` alone can run a real study
(EASE + Popularity + Random).
"""
import numpy as np

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


def _candidates(all_items, selected_items, filter_out_items):
    cand = np.setdiff1d(all_items, np.asarray(selected_items, dtype=int))
    return np.setdiff1d(cand, np.asarray(filter_out_items, dtype=int))


class Popularity(AlgorithmBase):
    """Non-personalized most-popular-items baseline (by interaction count)."""

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._all_items = self._ratings_df.item.unique()
        self._pop = None

    def fit(self):
        # popularity = number of interactions per item (a pandas Series indexed by item)
        self._pop = self._ratings_df.item.value_counts()

    def predict(self, selected_items, filter_out_items, k):
        candidates = _candidates(self._all_items, selected_items, filter_out_items)
        ranked = sorted(candidates, key=lambda c: int(self._pop.get(c, 0)), reverse=True)
        return list(ranked[:k])

    @classmethod
    def name(cls):
        return "Popularity"

    @classmethod
    def parameters(cls):
        return []


class Random(AlgorithmBase):
    """Random-recommendation baseline (a lower bound / sanity check)."""

    def __init__(self, loader, seed=42, **kwargs):
        self._all_items = loader.ratings_df.item.unique()
        self._seed = int(seed)
        self._rng = None

    def fit(self):
        self._rng = np.random.default_rng(self._seed)

    def predict(self, selected_items, filter_out_items, k):
        candidates = _candidates(self._all_items, selected_items, filter_out_items)
        k = min(k, len(candidates))
        return self._rng.choice(candidates, size=k, replace=False).tolist()

    @classmethod
    def name(cls):
        return "Random"

    @classmethod
    def parameters(cls):
        return [
            Parameter("seed", ParameterType.INT, 42, help="Random seed for reproducibility."),
        ]
