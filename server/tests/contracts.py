"""Reusable test harness for custom EasyStudy components.

Students/researchers writing a custom **algorithm**, **data loader**, **preference
elicitation** or **evaluation metric** shouldn't have to spin up the web UI to know their
component behaves. This module provides:

* :class:`TinyDataLoader` — a fully in-memory, download-free ``DataLoaderBase`` (30 users x
  50 items) you can fit any algorithm against in milliseconds; and
* ``assert_*_contract`` helpers that check a component honours the base-class contract.

Use them from your own ``pytest`` file — see ``docs/testing-components.md``. Example::

    from tests.contracts import TinyDataLoader, assert_algorithm_contract
    from my_plugin.my_algo import MyAlgo

    def test_my_algo():
        assert_algorithm_contract(MyAlgo, TinyDataLoader(), {"positive_threshold": 1.0, "l2": 0.5})
"""
import numpy as np
import pandas as pd

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    DataLoaderBase,
    EvaluationMetricBase,
    Parameter,
    PreferenceElicitationBase,
)


class TinyDataLoader(DataLoaderBase):
    """A minimal, in-memory dataset for tests — no files, no network.

    Demonstrates the ``item_id`` vs ``item_index`` distinction on purpose: the items have
    *ids* ``100..149`` (as an underlying dataset might), while their zero-based *indices* are
    ``0..49`` (what NumPy math uses). ``ratings_df`` exposes the dense ``user`` / ``item``
    (index) / ``rating`` columns that fastcompare algorithms consume. Sized (30 users × 50
    items) so preference-elicitation sampling has enough items to draw from.
    """

    _N_USERS = 30
    _ITEM_IDS = list(range(100, 150))  # 50 non-zero-based ids -> indices 0..49
    _CATEGORIES = ["cat_a", "cat_b", "cat_c"]

    def __init__(self, **kwargs):
        rng = np.random.default_rng(0)
        n_items = len(self._ITEM_IDS)
        rows = []
        for user in range(self._N_USERS):
            # each user rates a deterministic-random subset, so CF has signal to learn
            liked = rng.choice(n_items, size=15, replace=False)
            for item_index in liked:
                rows.append((user, int(item_index), float(rng.integers(3, 6))))
        self._ratings = pd.DataFrame(rows, columns=["user", "item", "rating"])
        self._items = pd.DataFrame({
            "item_id": self._ITEM_IDS,
            "item": list(range(n_items)),
            "title": [f"Item {i}" for i in range(n_items)],
        })

    # --- id <-> index mapping (the crux of item_id vs item_index) ---------------
    def get_item_index(self, item_id):
        return self._ITEM_IDS.index(int(item_id))

    def get_item_id(self, item_index):
        return self._ITEM_IDS[int(item_index)]

    # --- data ------------------------------------------------------------------
    def load_data(self):
        pass  # everything is built in __init__

    @property
    def ratings_df(self):
        return self._ratings

    @property
    def items_df(self):
        return self._items

    @property
    def items_df_indexed(self):
        return self._items.set_index("item")

    @property
    def rating_matrix(self):
        return (self._ratings.pivot(index="user", columns="item", values="rating")
                .reindex(columns=range(len(self._ITEM_IDS))).fillna(0).values)

    @property
    def distance_matrix(self):
        n = len(self._ITEM_IDS)
        return np.ones((n, n)) - np.identity(n)

    # --- presentation ----------------------------------------------------------
    def get_item_id_image_url(self, item_id):
        return f"https://example.invalid/img/{item_id}.jpg"

    def get_item_index_image_url(self, item_index):
        return self.get_item_id_image_url(self.get_item_id(item_index))

    def get_item_index_description(self, item_index):
        return f"Item {item_index}"

    def get_item_id_description(self, item_id):
        return self.get_item_index_description(self.get_item_index(item_id))

    def get_item_index_categories(self, item_index):
        return [self._CATEGORIES[int(item_index) % len(self._CATEGORIES)]]

    def get_all_categories(self):
        return list(self._CATEGORIES)

    @classmethod
    def name(cls):
        return "Tiny (in-memory test) dataset"

    @classmethod
    def parameters(cls):
        return []


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #

def _check_name_and_parameters(cls):
    assert callable(getattr(cls, "name", None)), f"{cls.__name__} must define name()"
    name = cls.name()
    assert isinstance(name, str) and name.strip(), f"{cls.__name__}.name() must be a non-empty str"
    params = cls.parameters()
    assert isinstance(params, list), f"{cls.__name__}.parameters() must return a list"
    for p in params:
        assert isinstance(p, Parameter), f"{cls.__name__}.parameters() items must be Parameter"


def assert_dataloader_contract(loader):
    """Assert a ``DataLoaderBase`` instance honours the contract."""
    assert isinstance(loader, DataLoaderBase)
    _check_name_and_parameters(type(loader))
    loader.load_data()

    df = loader.ratings_df
    assert {"user", "item"}.issubset(df.columns), "ratings_df needs 'user' and 'item' columns"
    assert len(df) > 0, "ratings_df must not be empty"

    # id <-> index must round-trip
    sample_index = int(df.item.iloc[0])
    item_id = loader.get_item_id(sample_index)
    assert loader.get_item_index(item_id) == sample_index, "get_item_id/get_item_index must round-trip"

    assert isinstance(loader.get_item_index_description(sample_index), str)
    assert isinstance(loader.get_item_id_description(item_id), str)
    assert isinstance(loader.get_item_id_image_url(item_id), str)
    assert hasattr(loader.get_all_categories(), "__iter__")


def assert_algorithm_contract(algo_cls, loader, params):
    """Fit ``algo_cls`` on ``loader`` and assert ``predict`` honours the contract."""
    assert issubclass(algo_cls, AlgorithmBase)
    _check_name_and_parameters(algo_cls)
    loader.load_data()

    algo = algo_cls(loader, **params)
    algo.fit()

    all_items = set(loader.ratings_df.item.unique())
    for k in (1, 3, 5):
        res = list(algo.predict([], [], k))
        assert len(res) == k, f"predict(k={k}) returned {len(res)} items"
        assert len(set(res)) == k, "predict must not return duplicates"
        assert set(res).issubset(all_items), "predict must return known item ids"

    # filter_out_items must never appear in the result
    filtered = list(all_items)[:3]
    res = list(algo.predict([], filtered, 3))
    assert set(res).isdisjoint(filtered), "predict must exclude filter_out_items"


def assert_elicitation_contract(elicitation_cls, loader, params):
    """Assert a ``PreferenceElicitationBase`` produces initial items to show."""
    assert issubclass(elicitation_cls, PreferenceElicitationBase)
    _check_name_and_parameters(elicitation_cls)
    loader.load_data()
    method = elicitation_cls(loader, **params)
    method.fit()
    data = method.get_initial_data()
    assert data is not None, "get_initial_data() must return something to show the user"


def assert_metric_contract(metric_cls, shown_items, selected_items):
    """Assert an ``EvaluationMetricBase`` returns a numeric result."""
    assert issubclass(metric_cls, EvaluationMetricBase)
    assert callable(getattr(metric_cls, "name", None))
    assert isinstance(metric_cls.name(), str) and metric_cls.name().strip()
    result = metric_cls().evaluate(shown_items, selected_items)
    assert isinstance(result, (int, float, np.integer, np.floating)), "evaluate() must return a number"
