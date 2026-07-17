"""Contract tests for the shipped components, using the download-free harness.

These run in CI without any dataset (unlike the MovieLens-gated fastcompare tests), so a
student can verify their own algorithm/loader the same way — see docs/testing-components.md.
"""
import pytest

from plugins.fastcompare.algo.ease import EASE
from plugins.fastcompare.algo.baselines import Popularity, Random

from tests.contracts import (
    TinyDataLoader,
    assert_algorithm_contract,
    assert_dataloader_contract,
)


def test_tiny_dataloader_contract():
    assert_dataloader_contract(TinyDataLoader())


def test_ease_algorithm_contract():
    assert_algorithm_contract(
        EASE, TinyDataLoader(), {"positive_threshold": 1.0, "l2": 0.5}
    )


# EASE + these two give the lightweight core enough algorithms to run a fastcompare
# comparison without the [lenskit]/[tensorflow] extras.
@pytest.mark.parametrize("algo_cls,params", [(Popularity, {}), (Random, {"seed": 1})])
def test_core_baseline_contracts(algo_cls, params):
    assert_algorithm_contract(algo_cls, TinyDataLoader(), params)
