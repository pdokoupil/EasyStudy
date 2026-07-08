"""Contract tests for the shipped components, using the download-free harness.

These run in CI without any dataset (unlike the MovieLens-gated fastcompare tests), so a
student can verify their own algorithm/loader the same way — see docs/testing-components.md.
"""
from plugins.fastcompare.algo.ease import EASE

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
