# Testing your components (no UI needed)

You should be able to know your custom **algorithm**, **data loader**, **preference elicitation** or
**metric** works *before* wiring it into a study and clicking through the web UI. EasyStudy ships a small
`pytest` harness for exactly that: [`server/tests/contracts.py`](https://github.com/pdokoupil/EasyStudy/blob/main/server/tests/contracts.py).

It gives you two things:

- **`TinyDataLoader`** — a fully in-memory `DataLoaderBase` (6 users × 8 items, no files, no network) you
  can fit any algorithm against in milliseconds.
- **`assert_*_contract` helpers** — one call that checks your component honours its base-class contract.

## Test a custom algorithm

```python
# server/tests/test_my_algo.py
from tests.contracts import TinyDataLoader, assert_algorithm_contract
from plugins.my_plugin.my_algo import MyAlgo

def test_my_algo_contract():
    assert_algorithm_contract(
        MyAlgo,
        TinyDataLoader(),
        {"positive_threshold": 1.0, "l2": 0.5},   # your algorithm's parameters()
    )
```

`assert_algorithm_contract` constructs your algorithm on the tiny dataset, calls `fit()`, then checks
`predict()`:

- returns exactly `k` items, with no duplicates, all from the known item space;
- never returns any `filter_out_items` (the framework relies on this to avoid re-showing items);
- exposes a non-empty unique `name()` and a `parameters()` list of `Parameter`.

## Test a custom data loader

```python
from tests.contracts import assert_dataloader_contract
from plugins.my_plugin.my_loader import MyLoader

def test_my_loader_contract():
    assert_dataloader_contract(MyLoader())
```

This checks `ratings_df` has the required dense `user` / `item` columns, that `get_item_id` /
`get_item_index` **round-trip** (the [item-id-vs-index](concepts.md#item-ids-vs-item-indices-and-users)
gotcha), and that the description/image/category helpers return sane types. Want to fit a *real*
algorithm against your loader too? Just pass it to `assert_algorithm_contract(EASE, MyLoader(), {...})`.

## Preference elicitation & metrics

```python
from tests.contracts import TinyDataLoader, assert_elicitation_contract, assert_metric_contract
from plugins.my_plugin.my_elicitation import MyElicitation
from plugins.my_plugin.my_metric import MyMetric

def test_elicitation():
    assert_elicitation_contract(MyElicitation, TinyDataLoader(), {})

def test_metric():
    assert_metric_contract(MyMetric, shown_items=[0, 1, 2, 3], selected_items=[1, 3])
```

## Running the tests

```bash
cd server
pytest -q                      # runs everything, including the contract tests
pytest tests/test_contracts.py # just the harness + shipped EASE
```

The contract tests need **no dataset download**, so they run everywhere (including CI). The heavier
`plugins/fastcompare/tests/test_algorithm.py` fits against real MovieLens and **skips** automatically if
you haven't run `python scripts/fetch_data.py --dataset ml-latest`.

!!! tip "Add your parameter combinations"
    `test_algorithm.py` has a `tested_algorithm_combinations` list — drop your `(AlgorithmClass, params)`
    there to also exercise it against real data once you're happy with the tiny-dataset contract.
