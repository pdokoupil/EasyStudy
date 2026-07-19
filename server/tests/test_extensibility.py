"""Extensibility / component tests that need no downloaded dataset (run in CI).

Covers the shipped evaluation metrics, the ml-latest-small demo loader (class-level, since
its data pipeline needs the CSVs), and the pure imdb_client URL/caching logic.
"""
import numpy as np
import pytest

from tests.contracts import TinyDataLoader, assert_metric_contract
from plugins.fastcompare.algo.metrics import nDCG, Precision, Count, ILD
from plugins.fastcompare.algo.algorithm_base import DataLoaderBase
from plugins.fastcompare.algo.wrappers.data_loadering import (
    MLLatestSmallDataLoader,
    MLDataLoaderWrapper,
)
from plugins.utils import imdb_client


# --- evaluation metrics -----------------------------------------------------------------

@pytest.mark.parametrize("metric_cls", [nDCG, Precision, Count])
def test_simple_metric_contracts(metric_cls):
    # shown = item indices presented, selected = the subset the user picked
    assert_metric_contract(metric_cls, shown_items=[0, 1, 2, 3], selected_items=[1, 3])


def test_ild_metric_numeric():
    loader = TinyDataLoader()
    loader.load_data()
    similarity = 1.0 - np.asarray(loader.distance_matrix)
    ild = ILD(loader.rating_matrix, similarity)
    val = ild.evaluate(shown_items=[0, 1, 2], selected_items=[0, 1])
    assert isinstance(val, (int, float, np.floating))
    assert val >= 0.0


# --- ml-latest-small demo loader (class-level; data pipeline needs CSVs) -----------------

def test_ml_latest_small_is_registered_dataloader():
    assert issubclass(MLLatestSmallDataLoader, MLDataLoaderWrapper)
    assert issubclass(MLLatestSmallDataLoader, DataLoaderBase)
    assert MLLatestSmallDataLoader.DATASET_DIR == "ml-latest-small"
    # distinct, non-empty display name; exposes the min_ratings_per_movie study-creation param
    assert MLLatestSmallDataLoader.name() and isinstance(MLLatestSmallDataLoader.name(), str)
    assert MLLatestSmallDataLoader.name() != MLDataLoaderWrapper.name()
    assert [p.name for p in MLLatestSmallDataLoader.parameters()] == ["min_ratings_per_movie"]


def test_ml_latest_small_uses_light_filters():
    # the big loader's heavy filters would empty the tiny catalog — the demo must override
    small = MLLatestSmallDataLoader._build_filters(MLLatestSmallDataLoader.__new__(MLLatestSmallDataLoader))
    big = MLDataLoaderWrapper._build_filters(MLDataLoaderWrapper.__new__(MLDataLoaderWrapper))
    assert len(small) < len(big)


# --- imdb_client: pure URL/caching logic (no network) -----------------------------------

def test_resize_amazon_rewrites_suffix():
    base = "https://m.media-amazon.com/images/M/ABC@"
    assert imdb_client._resize_amazon(base + "._V1_.jpg", 300) == base + "._V1_SX300.jpg"
    # already-suffixed URLs are re-sized, not doubled
    assert imdb_client._resize_amazon(base + "._V1_SY500_CR.jpg", 200) == base + "._V1_SX200.jpg"
    # URL with no recognizable suffix still gets a thumbnail suffix appended
    assert imdb_client._resize_amazon(base, 150) == base + "._V1_SX150.jpg"


def test_resize_amazon_leaves_non_amazon_untouched():
    tmdb = "https://image.tmdb.org/t/p/w342/poster.jpg"
    assert imdb_client._resize_amazon(tmdb, 300) == tmdb
    assert imdb_client._resize_amazon("", 300) == ""


def test_cover_lookup_is_cached(monkeypatch):
    calls = {"n": 0}

    class _FakeMovie:
        cover_url = "https://m.media-amazon.com/images/M/XYZ@._V1_.jpg"

    def _fake_get_movie(imdb_id):
        calls["n"] += 1
        return _FakeMovie()

    monkeypatch.setattr(imdb_client, "_get_movie", _fake_get_movie)
    imdb_client._cover_cache.clear()
    monkeypatch.delenv("TMDB_API_KEY", raising=False)

    u1 = imdb_client.get_cover_url(999999)
    u2 = imdb_client.get_cover_url(999999)
    assert u1 == u2 == "https://m.media-amazon.com/images/M/XYZ@._V1_SX300.jpg"
    assert calls["n"] == 1, "second lookup should hit the in-process cache, not re-fetch"
