# Extending fastcompare

`fastcompare` is extended by **subclassing a base class**. Put your class either directly in
`server/plugins/fastcompare/algo/*.py` or in a separate plugin folder
`server/plugins/<yourplugin>/*.py` — EasyStudy discovers subclasses automatically by walking the
`plugins/` tree. All base classes live in
`server/plugins/fastcompare/algo/algorithm_base.py`.

Every component implements `name()` (a unique display name) and `parameters()` (fields the researcher
can set in the study-creation UI). Parameters are declared with `Parameter(...)` and become clickable
inputs automatically.

!!! tip "Optional dependencies"
    If your component needs a heavy library (TensorFlow, LensKit, …), import it lazily / behind
    `try/except` and declare it as an extra in `pyproject.toml`. Discovery **skips** modules whose
    optional dependency is missing, so this keeps the lightweight core runnable.

## Algorithms

Subclass `AlgorithmBase` and implement `fit()`, `predict(selected_items, filter_out_items, k)`,
`name()`, `parameters()`. Worked example — the built-in NumPy **EASE** (no heavy deps):

```python
import numpy as np
import pandas as pd
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType

class EASE(AlgorithmBase):
    def __init__(self, loader, positive_threshold, l2, **kwargs):
        self._loader = loader
        self._all_items = loader.ratings_df.item.unique()
        self._rating_matrix = (
            loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0).values
        )
        self._threshold = positive_threshold
        self._l2 = l2
        self._items_count = self._rating_matrix.shape[1]
        self._weights = None

    def fit(self):                          # one-time training
        X = np.where(self._rating_matrix >= self._threshold, 1, 0).astype(np.float32)
        G = X.T @ X + self._l2 * np.identity(self._items_count, dtype=np.float32)
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self._weights = B

    def predict(self, selected_items, filter_out_items, k):   # per-participant recommendation
        candidates = np.setdiff1d(self._all_items, list(selected_items))
        candidates = np.setdiff1d(candidates, filter_out_items)
        if not selected_items:
            return np.random.choice(candidates, size=k, replace=False).tolist()
        user_vector = np.zeros((self._items_count,))
        for i in selected_items:
            user_vector[i] = 1.0
        preds = np.tensordot(user_vector.astype(np.float32), self._weights, axes=1)
        return [c for _, c in sorted(((preds[c], c) for c in candidates), reverse=True)][:k]

    @classmethod
    def name(cls):
        return "EASE"

    @classmethod
    def parameters(cls):
        return [
            Parameter("l2", ParameterType.FLOAT, 500, help="L2-norm regularization"),
            Parameter("positive_threshold", ParameterType.FLOAT, 2.5,
                      help="Threshold to binarize ratings into positive/negative."),
        ]
```

`predict` receives the participant's `selected_items`, items to `filter_out` (already seen), and `k`
(how many to return). It must return `k` item ids, none of them filtered out.

Existing wrappers (`plugins/fastcompare/algo/wrappers/`) show how to wrap LensKit and TF-Recommenders
models — a good reference if your algorithm needs an extra.

## Datasets

Subclass `DataLoaderBase`. In addition to `name()`/`parameters()`, implement `load_data()` and the
accessors EasyStudy uses to render items and compute metrics:

- Data frames / matrices: `ratings_df`, `items_df`, `items_df_indexed`, `rating_matrix`,
  `distance_matrix`.
- Item lookups: `get_item_index(item_id)`, `get_item_id(item_index)`,
  `get_item_index_description(...)` / `get_item_id_description(...)`,
  `get_item_index_categories(...)`, `get_all_categories()`.
- Images: `get_item_id_image_url(item_id)` / `get_item_index_image_url(item_index)`.

Get inspired by `GoodbooksDataLoader` / `MLDataLoaderWrapper` in `plugins/utils/`.

For images, return a static URL:
```python
from flask import url_for
url_for('static', filename=f'datasets/goodbooks-10k/img/{item_id}.jpg')
```
Put your dataset under [`server/static/datasets/`](https://github.com/pdokoupil/EasyStudy/tree/main/server/static/datasets).
If your deployment only lets you write inside `plugins/`, ship assets in your plugin and address them
with `pm.emit_assets('yourPluginName', filename)` (import `pm` from `app`, and check
`has_app_context()` first). Returning remote `http://…` URLs works but slows the study down.

## Preference elicitation

Subclass `PreferenceElicitationBase` (implement `fit()`, the elicitation logic, `name()`,
`parameters()`). Popularity sampling (`plugins/utils/popularity_sampling.py`) is a dependency-free
reference; the TFRS-based method requires `easystudy[tensorflow]`.

## Metrics

Subclass `EvaluationMetricBase` and implement `evaluate(shown_items, selected_items)` + `name()`. The
built-ins are nDCG@K, Precision@K, ILD, and Selection Count — new metrics appear automatically in the
study **Results** view.

## Testing your component

Add a test under `server/plugins/fastcompare/tests/` (or your plugin). See
`test_algorithm.py` for the fixture pattern (it skips cleanly if the dataset isn't downloaded). Run:

```bash
cd server && pytest -q
```
