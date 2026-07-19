"""End-to-end tests for everything triggered during study creation, on a tiny in-memory
dataset (no downloads). Covers: building a data loader, building + fitting each shipped
algorithm and producing recommendations, and building + running each preference-elicitation
method. Parameters are taken from each component's own declared defaults, so new components
are covered automatically.

RecBole models are covered too, but only when the `recbole` extra is installed (they need
torch); otherwise those cases are skipped.
"""
import numpy as np
import pytest

from tests.contracts import TinyDataLoader, assert_elicitation_contract
from plugins.fastcompare.algo.ease import EASE
from plugins.fastcompare.algo.baselines import Popularity, Random
from plugins.fastcompare.algo.wrappers.elicitation import (
    PopularitySamplingElicitationWrapper,
    PopularitySamplingFromBucketsElicitationWrapper,
    MultiObjectiveSamplingFromBucketsElicitationWrapper,
)

CORE_ALGORITHMS = [EASE, Popularity, Random]

ELICITATIONS = [
    PopularitySamplingElicitationWrapper,
    PopularitySamplingFromBucketsElicitationWrapper,
    MultiObjectiveSamplingFromBucketsElicitationWrapper,
]


def _defaults(cls):
    """Build a kwargs dict from a component's declared Parameter defaults."""
    return {p.name: p.default for p in cls.parameters()}


def _loader():
    dl = TinyDataLoader()
    dl.load_data()
    return dl


# --- data loader creation ---------------------------------------------------------------

def test_dataloader_creation_and_shapes():
    dl = _loader()
    assert len(dl.items_df) > 0
    rm = np.asarray(dl.rating_matrix)
    dm = np.asarray(dl.distance_matrix)
    assert rm.shape[1] == dm.shape[0] == dm.shape[1]  # items align across matrices


# --- algorithm creation -> fit -> predict -----------------------------------------------

@pytest.mark.parametrize("algo_cls", CORE_ALGORITHMS, ids=lambda c: c.name())
def test_algorithm_create_fit_predict(algo_cls):
    dl = _loader()
    algo = algo_cls(dl, **_defaults(algo_cls))   # create with its own default params
    algo.fit()                                   # train
    selected, filtered, k = [0, 1], [2], 3
    rec = algo.predict(selected, filtered, k)    # recommend for a cold user
    assert isinstance(rec, list)
    assert len(rec) <= k
    assert not (set(rec) & set(selected)), "must not recommend already-selected items"
    assert not (set(rec) & set(filtered)), "must not recommend filtered-out items"
    all_items = set(int(i) for i in dl.ratings_df.item.unique())
    assert set(rec) <= all_items, "recommendations must be valid item indices"


# --- preference elicitation creation -> fit -> initial data -----------------------------

@pytest.mark.parametrize("elic_cls", ELICITATIONS, ids=lambda c: c.name())
def test_elicitation_create_fit_initial_data(elic_cls):
    assert_elicitation_contract(elic_cls, _loader(), _defaults(elic_cls))


# --- RecBole models (only when the extra is installed) ----------------------------------

try:
    import torch  # noqa: F401
    from plugins.recbole.algorithms import BPR, LightGCN, NGCF, NeuMF, DMF
    _RECBOLE_MODELS = [BPR, LightGCN, NGCF, NeuMF, DMF]
except Exception:
    _RECBOLE_MODELS = []


@pytest.mark.skipif(not _RECBOLE_MODELS, reason="recbole extra not installed")
@pytest.mark.parametrize("model_cls", _RECBOLE_MODELS, ids=[c.name() for c in _RECBOLE_MODELS])
def test_recbole_model_create_fit_predict(model_cls):
    dl = _loader()
    params = _defaults(model_cls)
    params["epochs"] = 1                          # keep the test fast
    algo = model_cls(dl, **params)
    algo.fit()                                    # exercises training incl. the scipy dok patch
    rec = algo.predict([0, 1], [2], 3)
    assert isinstance(rec, list) and len(rec) <= 3
    assert not (set(rec) & {0, 1, 2})
